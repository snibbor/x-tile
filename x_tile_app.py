###########################
# Filename: x_tile_app.py #
###########################
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import base64
import io
import json
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import median_survival_times


# For parallel processing and background callbacks:
import concurrent.futures
import os
from dash import DiskcacheManager, CeleryManager

if "REDIS_URL" in os.environ:
    from celery import Celery

    celery_app = Celery(__name__, broker=os.environ["REDIS_URL"], backend=os.environ["REDIS_URL"])
    background_callback_manager = CeleryManager(celery_app)
else:
    import diskcache

    cache = diskcache.Cache("./cache")
    background_callback_manager = DiskcacheManager(cache)


##############################################
# 1) DATA HANDLING & HELPER FUNCTIONS
##############################################
def assign_groups(values, low_cut, high_cut):
    """
    For a binary split (low_cut == high_cut):
      - group 'low' if value < low_cut
      - group 'high' if value >= low_cut
    For a two-cutpoint split (low_cut < high_cut):
      - group 'low' if value < low_cut
      - group 'mid' if low_cut <= value < high_cut
      - group 'high' if value >= high_cut
    """
    if low_cut == high_cut:
        return pd.cut(values, bins=[-np.inf, low_cut, np.inf], labels=["low", "high"])
    else:
        return pd.cut(values, bins=[-np.inf, low_cut, high_cut, np.inf], labels=["low", "mid", "high"])


def compute_logrank_statistic(data, low_cut, high_cut):
    """Compute chi-square (log-rank test statistic) for the given split."""
    df = data.copy()
    df["group"] = assign_groups(df["biomarker"], low_cut, high_cut)
    if df["group"].value_counts().min() < 5:
        return np.nan
    res = multivariate_logrank_test(df["time"], df["group"], df["event"])
    return res.test_statistic


def compute_direction(data, low_cut, high_cut):
    """
    Determine the direction:
      +1 if the 'high' group has longer median survival (direct association),
      -1 if the 'high' group has shorter median survival (inverse association),
       0 if undefined.
    For a three-category split, we compare the extremes.
    """
    df = data.copy()
    df["group"] = assign_groups(df["biomarker"], low_cut, high_cut)
    try:
        low_median = df.loc[df["group"] == "low", "time"].median()
        high_median = df.loc[df["group"] == "high", "time"].median()
    except Exception:
        return 0
    if pd.isna(low_median) or pd.isna(high_median):
        return 0
    if high_median < low_median:
        return -1
    elif high_median > low_median:
        return 1
    else:
        return 0


def parse_uploaded_file(contents, filename):
    """Parse an uploaded CSV or Excel file. Expect columns: 'biomarker', 'time', 'event'."""
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename.lower():
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            return df
        elif any(ext in filename.lower() for ext in ["xls", "xlsx"]):
            df = pd.read_excel(io.BytesIO(decoded))
            return df
    except Exception as e:
        print("Error parsing file:", e)
        return None
    return None


##############################################
# 2) DEFAULT DATA (for testing/debugging)
##############################################
np.random.seed(42)
N = 300  # Increase sample size for more robust splits
# Simulate a binary mixture:
#   - Group 0: biomarker ~ N(40, 5)
#   - Group 1: biomarker ~ N(60, 5)
group = np.random.binomial(1, 0.5, N)
_debug_biomarker = np.where(group == 0, np.random.normal(40, 5, N), np.random.normal(60, 5, N))
# Hazard rate: lower if biomarker < 50, higher if biomarker >= 50.
_debug_hazard = np.where(_debug_biomarker < 50, 0.03, 0.07)
_debug_survival = np.random.exponential(scale=1 / _debug_hazard, size=N)
# Simulate event occurrence with a fairly high event rate (e.g., 80% chance)
_debug_events = np.random.binomial(1, 0.8, N)
_debug_df = pd.DataFrame({"biomarker": _debug_biomarker, "time": _debug_survival, "event": _debug_events})
default_data_json = _debug_df.to_json(date_format="iso", orient="split")


##############################################
# 3) CANDIDATE GRID FOR THE HEATMAP (Precomputed)
##############################################
def build_candidate_grid(data):
    """
    Build a grid of candidate cutpoints.
    Orientation:
      - X axis: low cutpoints (ascending)
      - Y axis: high cutpoints (top = highest, bottom = lowest)
    For each valid candidate (low <= high), compute the signed statistic = chi-square * direction.
    """
    df = data.copy()
    pct_vals = np.percentile(df["biomarker"], np.linspace(5, 95, 25))
    pct_vals = np.unique(np.round(pct_vals, 2))
    low_array = np.sort(pct_vals)  # For X axis (ascending)
    high_array_asc = np.sort(pct_vals)  # Then reverse for Y axis
    n_low = len(low_array)
    n_high = len(high_array_asc)
    M = np.full((n_high, n_low), np.nan)

    for i in range(n_high):
        high_cut = high_array_asc[i]
        for j in range(n_low):
            low_cut = low_array[j]
            if low_cut <= high_cut:
                chi2 = compute_logrank_statistic(df, low_cut, high_cut)
                if not np.isnan(chi2):
                    sign = compute_direction(df, low_cut, high_cut)
                    M[i, j] = chi2 * sign
    high_array = high_array_asc[::-1]  # Reverse: largest high cut at the top
    M = np.flipud(M)
    return low_array, high_array, M


##############################################
# 4) FIGURE HELPER FUNCTIONS
##############################################
def make_heatmap_figure(low_vals, high_vals, matrix):
    """Create a heatmap with X = low cuts, Y = high cuts (largest at top)."""
    z_max = np.nanmax(np.abs(matrix))
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=low_vals,
            y=high_vals,
            colorscale="RdYlGn",
            zmin=-z_max,
            zmax=z_max,
            zmid=0,
            hovertemplate=("Low cut: %{x}<br>" "High cut: %{y}<br>" "Score: %{z:.2f}<extra></extra>"),
        )
    )
    fig.update_layout(
        title="X-Tile Heatmap (Signed Chi²)", xaxis_title="Low Cut", yaxis_title="High Cut", template="plotly_white"
    )
    return fig


def km_figure(data, low_cut, high_cut):
    """Generate Kaplan–Meier curves using the selected cutpoints."""
    df = data.copy()
    df["group"] = assign_groups(df["biomarker"], low_cut, high_cut)
    if df["group"].value_counts().min() < 1:
        return go.Figure()
    kmf = KaplanMeierFitter()
    fig = go.Figure()
    for grp in df["group"].unique():
        subset = df[df["group"] == grp]
        if len(subset) == 0:
            continue
        kmf.fit(subset["time"], subset["event"], label=str(grp))
        col_name = kmf.survival_function_.columns[0]
        fig.add_trace(
            go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_[col_name], mode="lines", name=str(grp))
        )
    fig.update_layout(
        title="Kaplan–Meier Curves", xaxis_title="Time", yaxis_title="Survival Probability", template="plotly_white"
    )
    return fig


def hist_figure(data, low_cut, high_cut):
    """Plot histogram of biomarker values with vertical cut lines."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data["biomarker"], nbinsx=20, marker_color="lightblue"))
    fig.add_vline(x=low_cut, line_width=2, line_dash="dash", line_color="red")
    fig.add_vline(x=high_cut, line_width=2, line_dash="dash", line_color="green")
    fig.update_layout(
        title="Biomarker Distribution", xaxis_title="Biomarker", yaxis_title="Count", template="plotly_white"
    )
    return fig


##############################################
# 5) CROSS-VALIDATION SPLITTING FUNCTIONS
##############################################
# Option: Simple Random Split
def random_split(data, test_size=0.5, random_state=None):
    from sklearn.model_selection import train_test_split

    train, val = train_test_split(data, test_size=test_size, random_state=random_state)
    return train, val


# Option: Stratified Split by Event
def stratified_split_event(data, test_size=0.5, random_state=None):
    from sklearn.model_selection import StratifiedShuffleSplit

    X = data.index.values.reshape(-1, 1)  # dummy variable
    y = data["event"].values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_idx, val_idx in sss.split(X, y):
        train = data.iloc[train_idx]
        val = data.iloc[val_idx]
    return train, val


# Option: Stratified Split by Time (using quantile bins)
def stratified_split_time(data, test_size=0.5, n_bins=4, random_state=None):
    from sklearn.model_selection import StratifiedShuffleSplit

    df = data.copy()
    try:
        df["time_bin"] = pd.qcut(df["time"], q=n_bins, duplicates="drop")
    except Exception:
        df["time_bin"] = pd.cut(df["time"], bins=n_bins)
    X = df.index.values.reshape(-1, 1)
    y = df["time_bin"].astype(str).values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_idx, val_idx in sss.split(X, y):
        train = df.iloc[train_idx].drop(columns=["time_bin"])
        val = df.iloc[val_idx].drop(columns=["time_bin"])
    return train, val


def single_cv_iteration(args):
    """
    A single cross-validation iteration.
    Input: tuple (data, category_type, split_method, test_size)
    Returns a dict with:
      - training_cutpoints, training_score
      - validation_score, validation_p
      - validation_median_survival: dict with group->median survival
      - validation_hr: dict with HR metrics (per dummy variable)
    """
    data, category_type, split_method, test_size = args
    # Use the selected splitting method.
    if split_method == "random":
        train, val = random_split(data, test_size=test_size)
    elif split_method == "stratified_event":
        train, val = stratified_split_event(data, test_size=test_size)
    elif split_method == "stratified_time":
        train, val = stratified_split_time(data, test_size=test_size, n_bins=4)
    else:
        train, val = random_split(data, test_size=test_size)

    best_score = -np.inf
    best_cutpoints = None
    if category_type == "3-category":
        low_vals, high_vals, _ = build_candidate_grid(train)
        for low in low_vals:
            for high in high_vals:
                if low < high:
                    score = compute_logrank_statistic(train, low, high)
                    if score is not None and not np.isnan(score) and score > best_score:
                        best_score = score
                        best_cutpoints = (low, high)
    elif category_type == "2-category":
        candidate_vals = np.sort(np.unique(np.percentile(train["biomarker"], np.linspace(5, 95, 25))))
        for cv in candidate_vals:
            score = compute_logrank_statistic(train, cv, cv)
            if score is not None and not np.isnan(score) and score > best_score:
                best_score = score
                best_cutpoints = (cv, cv)
    if best_cutpoints is None:
        return None

    # Calculate validation log-rank test p-value and score.
    groups_val = assign_groups(val["biomarker"], best_cutpoints[0], best_cutpoints[1])
    res = multivariate_logrank_test(val["time"], groups_val, val["event"])
    p_val = res.p_value
    val_score = compute_logrank_statistic(val, best_cutpoints[0], best_cutpoints[1])

    # Compute median survival per group from the validation set.
    km_results = {}
    for grp in groups_val.dropna().unique():
        subset = val[groups_val == grp]
        if len(subset) == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(subset["time"], event_observed=subset["event"])
        km_results[grp] = kmf.median_survival_time_

    # Compute hazard ratios via a univariate Cox model.
    val_df = val.copy()
    val_df["group"] = groups_val
    # Ensure group is categorical, ordering by sorted group names.
    # the order should always be low < mid < high.
    # If only 2 groups, order by low < high.
    if len(val_df["group"].unique()) == 2:
        val_df["group"] = pd.Categorical(val_df["group"], categories=["low", "high"], ordered=True)
    else:
        val_df["group"] = pd.Categorical(val_df["group"], categories=["low", "mid", "high"], ordered=True)

    dummies = pd.get_dummies(val_df["group"], prefix="grp", drop_first=True)
    cox_data = pd.concat([val_df[["time", "event"]], dummies], axis=1)
    from lifelines import CoxPHFitter

    try:
        cph = CoxPHFitter()
        # Use all dummy columns (if any)
        if dummies.shape[1] > 0:
            formula = " + ".join(dummies.columns)
            cph.fit(cox_data, duration_col="time", event_col="event", formula=formula)
        else:
            # No dummy if only one group exists.
            cph.fit(cox_data, duration_col="time", event_col="event")
        hr_summary = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].to_dict(
            orient="index"
        )
    except Exception:
        hr_summary = {}

    return {
        "training_cutpoints": best_cutpoints,
        "training_score": best_score,
        "validation_score": val_score,
        "validation_p": p_val,
        "validation_median_survival": km_results,
        "validation_hr": hr_summary,
    }


def cross_validate_cutpoints(data, num_iter=20, category_type="3-category", split_method="random", test_size=0.5):
    """
    Run Monte Carlo cross-validation in parallel.
    Returns a list of iteration results.
    """
    args = [(data, category_type, split_method, test_size)] * num_iter
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(single_cv_iteration, args))
    results = [r for r in results if r is not None]
    return results


##############################################
# 6) DASH APP LAYOUT
##############################################
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container(
    [
        html.H1("Jack's X-Tile", style={"marginTop": 20}),
        # Upload component
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select a CSV/XSLX File")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=False,
        ),
        dcc.Store(id="dataset-json", data=default_data_json),
        dcc.Store(id="column-mapping", data={}),
        dcc.Store(id="candidate-grid-store"),
        # Container for column mapping dropdowns (initially hidden)
        html.Div(
            [
                html.H4("Select Column Mapping"),
                html.Div(
                    [html.Label("Biomarker Column:"), dcc.Dropdown(id="col-biomarker", options=[], multi=False)],
                    style={"margin": "10px"},
                ),
                html.Div(
                    [html.Label("Time Column:"), dcc.Dropdown(id="col-time", options=[], multi=False)],
                    style={"margin": "10px"},
                ),
                html.Div(
                    [html.Label("Event Column:"), dcc.Dropdown(id="col-event", options=[], multi=False)],
                    style={"margin": "10px"},
                ),
                dbc.Button("Confirm Column Mapping", id="confirm-mapping", color="primary"),
            ],
            id="column-mapping-div",
            style={"display": "none"},
        ),
        dcc.Graph(id="heatmap-figure", config={"displaylogo": False}),
        html.Div(id="chosen-cutpoints", style={"margin": "20px 0", "fontWeight": "bold"}),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="km-figure", config={"displaylogo": False}), md=6),
                dbc.Col(dcc.Graph(id="hist-figure", config={"displaylogo": False}), md=6),
            ]
        ),
        html.Hr(),
        html.H3("Monte Carlo Cross-Validation"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Cutpoint Type:"),
                        dcc.Dropdown(
                            id="cv-category",
                            options=[
                                {"label": "3-Category Split", "value": "3-category"},
                                {"label": "2-Category (Binary) Split", "value": "2-category"},
                            ],
                            value="3-category",
                            clearable=False,
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Number of Iterations:"),
                        dcc.Input(id="cv-iterations", type="number", value=20, min=5, step=5),
                    ],
                    md=2,
                ),
                dbc.Col(
                    [
                        html.Label("Val. Set Size:"),
                        dcc.Input(id="cv-val-size", type="number", value=0.5, min=0.1, max=0.9, step=0.1),
                    ],
                    md=2,
                ),
                dbc.Col(
                    [
                        html.Label("Split Method:"),
                        dcc.Dropdown(
                            id="cv-split-method",
                            options=[
                                {"label": "Random Split", "value": "random"},
                                {"label": "Stratified by Event", "value": "stratified_event"},
                                {"label": "Stratified by Time", "value": "stratified_time"},
                            ],
                            value="random",
                            clearable=False,
                        ),
                    ],
                    md=3,
                ),
                dbc.Col([dbc.Button("Run Cross-Validation", id="cv-button", color="primary")], md=2),
            ],
            align="center",
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(html.Div(id="cv-output"), md=6),
                dbc.Col(dcc.Graph(id="cv-histogram", config={"displaylogo": False}), md=6),
            ]
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(html.Div(id="cv-stats-output"), md=6),
                dbc.Col(html.Div(id="total-stats-output"), md=6),
            ]
        ),
        dbc.Row([dbc.Col(dcc.Graph(id="km-total", config={"displaylogo": False}), md=6)]),
    ],
    fluid=True,
)


##############################################
# 7) DASH CALLBACKS
##############################################
@app.callback(
    [
        Output("col-biomarker", "options"),
        Output("col-time", "options"),
        Output("col-event", "options"),
        Output("column-mapping-div", "style"),
    ],
    Input("dataset-json", "data"),
)
def update_column_mapping_options(data_json):
    data = pd.read_json(data_json, orient="split")
    options = [{"label": col, "value": col} for col in data.columns]
    # If there are more than 3 columns, show the mapping div.
    style = {"display": "block"} if len(data.columns) > 3 else {"display": "none"}
    return options, options, options, style


# Callback to save the user’s selected column mapping.
@app.callback(
    Output("column-mapping", "data"),
    Input("confirm-mapping", "n_clicks"),
    [State("col-biomarker", "value"), State("col-time", "value"), State("col-event", "value")],
    prevent_initial_call=True,
)
def save_column_mapping(n_clicks, biomarker_col, time_col, event_col):
    mapping = {"biomarker": biomarker_col, "time": time_col, "event": event_col}
    return mapping


# A helper function that applies the column mapping (if it exists) to the dataframe.
def apply_column_mapping(df, mapping):
    """
    If a mapping (a dict with keys "biomarker", "time", "event") is provided and the values exist
    as columns in the dataframe, then rename those columns to the standard names.
    Otherwise, return the original dataframe.
    """
    if mapping and isinstance(mapping, dict):
        # Only rename if all three mapping values are provided and exist in df.columns.
        if (
            mapping.get("biomarker") in df.columns
            and mapping.get("time") in df.columns
            and mapping.get("event") in df.columns
        ):
            new_names = {
                mapping.get("biomarker"): "biomarker",
                mapping.get("time"): "time",
                mapping.get("event"): "event",
            }
            df = df.rename(columns=new_names)
    return df


# Override the earlier upload callback so that when a file is uploaded,
# we simply store the full DataFrame.
@app.callback(Output("dataset-json", "data"), Input("upload-data", "contents"), State("upload-data", "filename"))
def handle_upload(contents, filename):
    if contents is not None:
        df = parse_uploaded_file(contents, filename)
        if df is not None:
            return df.to_json(date_format="iso", orient="split")
    return default_data_json


@app.callback(
    [Output("candidate-grid-store", "data"), Output("heatmap-figure", "figure")],
    [Input("dataset-json", "data"), Input("confirm-mapping", "n_clicks")],
    State("column-mapping", "data"),
)
def update_heatmap_and_store(data_json, confirm_clicks, mapping):
    data = pd.read_json(data_json, orient="split")
    # Apply mapping if provided; if mapping is missing or invalid, the function returns data unchanged.
    data = apply_column_mapping(data, mapping)
    try:
        low_vals, high_vals, matrix = build_candidate_grid(data)
    except Exception as e:
        # If an error still occurs, return an empty figure with an error message.
        print("Error building candidate grid:", e)
        return {}, go.Figure(data=[go.Scatter(text=["Error building heatmap. Check column mapping."], mode="text")])
    fig = make_heatmap_figure(low_vals, high_vals, matrix)
    store_dict = {"low_vals": list(low_vals), "high_vals": list(high_vals), "matrix": matrix.tolist()}
    return store_dict, fig


@app.callback(
    [Output("chosen-cutpoints", "children"), Output("km-figure", "figure"), Output("hist-figure", "figure")],
    Input("heatmap-figure", "clickData"),
    State("dataset-json", "data"),
    State("column-mapping", "data"),
)
def update_plots_from_click(clickData, data_json, mapping):
    chosen_text = "Click the heatmap to pick cutpoints."
    km_fig = go.Figure()
    hist_fig = go.Figure()
    data = pd.read_json(data_json, orient="split")
    data = apply_column_mapping(data, mapping)
    if clickData:
        low_cut = float(clickData["points"][0]["x"])
        high_cut = float(clickData["points"][0]["y"])
        if low_cut > high_cut:
            chosen_text = f"Invalid selection: low={low_cut} > high={high_cut}"
        else:
            chosen_text = f"Chosen cutpoints: low={low_cut:.2f}, high={high_cut:.2f}"
            km_fig = km_figure(data, low_cut, high_cut)
            hist_fig = hist_figure(data, low_cut, high_cut)
    return chosen_text, km_fig, hist_fig


@app.callback(
    output=[
        Output("cv-output", "children"),
        Output("cv-histogram", "figure"),
        Output("cv-stats-output", "children"),
        Output("total-stats-output", "children"),
        Output("km-total", "figure"),
    ],
    inputs=[Input("cv-button", "n_clicks")],
    state=[
        State("cv-iterations", "value"),
        State("cv-category", "value"),
        State("cv-split-method", "value"),
        State("cv-val-size", "value"),
        State("dataset-json", "data"),
        State("column-mapping", "data"),
    ],
    manager=background_callback_manager,
    prevent_initial_call=True,
    background=True,
)
def run_cross_validation(n_clicks, iterations, cv_category, cv_split_method, test_size, data_json, mapping):
    """
    Runs Monte Carlo cross-validation in the background.
    Aggregates the optimal cutpoints, validation log-rank p-values,
    group median survivals, and hazard ratios (and their p-values)
    from each CV iteration.
    Then, using the aggregated (median) optimal cutpoints, it computes:
      - Median survival per group and hazard ratios (and p-values) for the TOTAL COHORT.
    Finally, it produces a KM survival curve for the total cohort.
    """
    data = pd.read_json(data_json, orient="split")
    data = apply_column_mapping(data, mapping)

    # Run cross-validation and aggregate CV iteration results.
    cv_results = cross_validate_cutpoints(
        data,
        num_iter=iterations,
        category_type=cv_category,
        split_method=cv_split_method,
        test_size=test_size,
    )
    if len(cv_results) == 0:
        return (
            "No valid cutpoints found in cross-validation.",
            go.Figure(),
            "No CV stats available.",
            "No total cohort stats available.",
            go.Figure(),
        )
    # Aggregate training cutpoints and validation p-values.
    low_list = [r["training_cutpoints"][0] for r in cv_results]
    high_list = [r["training_cutpoints"][1] for r in cv_results]
    p_list = [r["validation_p"] for r in cv_results if r["validation_p"] is not None]

    low_median = np.median(low_list)
    low_lower = np.percentile(low_list, 2.5)
    low_upper = np.percentile(low_list, 97.5)

    high_median = np.median(high_list)
    high_lower = np.percentile(high_list, 2.5)
    high_upper = np.percentile(high_list, 97.5)

    p_median = np.median(p_list)
    p_lower = np.percentile(p_list, 2.5)
    p_upper = np.percentile(p_list, 97.5)

    table_style = {
        "width": "100%",
        "borderCollapse": "separate",
        "borderSpacing": "40px 30px",  # horizontal, vertical spacing
        "margin": "10px 0",
    }

    cv_table = html.Table(
        [
            html.Thead(
                html.Tr([html.Th("Parameter"), html.Th("Median"), html.Th("Lower 95% CI"), html.Th("Upper 95% CI")])
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td("Low Cutpoint"),
                            html.Td(f"{low_median:.2f}"),
                            html.Td(f"{low_lower:.2f}"),
                            html.Td(f"{low_upper:.2f}"),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("High Cutpoint"),
                            html.Td(f"{high_median:.2f}"),
                            html.Td(f"{high_lower:.2f}"),
                            html.Td(f"{high_upper:.2f}"),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("Log-Rank p-value"),
                            html.Td(f"{p_median:.4f}"),
                            html.Td(f"{p_lower:.4f}"),
                            html.Td(f"{p_upper:.4f}"),
                        ]
                    ),
                ]
            ),
        ],
        style=table_style,
    )
    cv_summary = html.Div([html.H4("Cross-Validation Summary"), cv_table])
    hist_fig = go.Figure(data=go.Histogram(x=p_list, nbinsx=min(10, len(p_list)), marker_color="orange"))
    hist_fig.update_layout(
        title="Distribution of Validation p-values",
        xaxis_title="p-value",
        yaxis_title="Frequency",
        template="plotly_white",
    )

    # Aggregate median survival per group from CV iterations.
    surv_agg = {}
    for r in cv_results:
        for grp, med in r["validation_median_survival"].items():
            surv_agg.setdefault(grp, []).append(med)
    surv_rows = []
    for grp in sorted(surv_agg.keys()):
        arr = np.array(surv_agg[grp])
        med = np.median(arr)
        lower = np.percentile(arr, 2.5)
        upper = np.percentile(arr, 97.5)
        surv_rows.append(
            html.Tr([html.Td(str(grp)), html.Td(f"{med:.2f}"), html.Td(f"{lower:.2f}"), html.Td(f"{upper:.2f}")])
        )
    surv_table = html.Table(
        [
            html.Thead(
                html.Tr(
                    [html.Th("Group"), html.Th("Median Survival"), html.Th("Lower 95% CI"), html.Th("Upper 95% CI")]
                )
            ),
            html.Tbody(surv_rows),
        ],
        style=table_style,
    )

    # Aggregate hazard ratios and their p-values over CV iterations.
    hr_agg = {}
    hr_p_agg = {}
    for r in cv_results:
        hr_dict = r.get("validation_hr", {})
        for key, metrics in hr_dict.items():
            hr_agg.setdefault(key, []).append(metrics["exp(coef)"])
            hr_p_agg.setdefault(key, []).append(metrics["p"])
    hr_rows = []
    for key in sorted(hr_agg.keys()):
        arr = np.array(hr_agg[key])
        med = np.median(arr)
        lower = np.percentile(arr, 2.5)
        upper = np.percentile(arr, 97.5)
        p_arr = np.array(hr_p_agg[key])
        p_med = np.median(p_arr)
        p_lower = np.percentile(p_arr, 2.5)
        p_upper = np.percentile(p_arr, 97.5)
        comp = key.replace("grp_", "").capitalize()
        hr_rows.append(
            html.Tr(
                [
                    html.Td(comp),
                    html.Td(f"{med:.2f}"),
                    html.Td(f"{lower:.2f}"),
                    html.Td(f"{upper:.2f}"),
                    html.Td(f"{p_med:.4f}"),
                ]
            )
        )
    hr_table = html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Comparison vs Low"),
                        html.Th("Hazard Ratio"),
                        html.Th("Lower 95% CI"),
                        html.Th("Upper 95% CI"),
                        html.Th("p-value"),
                    ]
                )
            ),
            html.Tbody(hr_rows),
        ],
        style=table_style,
    )
    cv_stats = html.Div(
        [
            html.H4("Validation Set Statistics (aggregated over CV runs)"),
            html.H5("Median Survival per Group"),
            surv_table,
            html.H5("Hazard Ratios (vs. Lowest Group)"),
            hr_table,
        ]
    )

    # Now compute TOTAL COHORT STATISTICS using the aggregated optimal cutpoints.
    if cv_category == "2-category":
        final_cut = (low_median, low_median)
    else:
        final_cut = (low_median, high_median)
    total_groups = assign_groups(data["biomarker"], final_cut[0], final_cut[1])

    # Compute median survival per group in total cohort along with 95% CI using lifelines' median_survival_times.
    tot_surv = {}
    for grp in total_groups.dropna().unique():
        sub = data[total_groups == grp]
        if len(sub) == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub["time"], event_observed=sub["event"])
        # Use the built-in lifelines function to derive the median's CI.
        ci_df = median_survival_times(kmf.confidence_interval_)
        # Extract the lower and upper CI from the resulting DataFrame.
        # (Assumes that the column names contain '_lower_0.95' and '_upper_0.95', as in your old code.)
        lower_col = [c for c in ci_df.columns if "_lower_0.95" in c][0]
        upper_col = [c for c in ci_df.columns if "_upper_0.95" in c][0]
        tot_surv[grp] = (kmf.median_survival_time_, ci_df.iloc[0][lower_col], ci_df.iloc[0][upper_col])

    tot_surv_rows = []
    for grp in sorted(tot_surv.keys()):
        median_val, lower_ci, upper_ci = tot_surv[grp]
        tot_surv_rows.append(
            html.Tr(
                [
                    html.Td(str(grp)),
                    html.Td(f"{median_val:.2f}"),
                    html.Td(f"{lower_ci:.2f}"),
                    html.Td(f"{upper_ci:.2f}"),
                ]
            )
        )
    tot_surv_table = html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Group"),
                        html.Th("Median Survival"),
                        html.Th("Lower 95% CI"),
                        html.Th("Upper 95% CI"),
                    ]
                )
            ),
            html.Tbody(tot_surv_rows),
        ],
        style=table_style,
    )

    # Compute hazard ratios for total cohort via a Cox model.
    total_df = data.copy()
    total_df["group"] = total_groups
    if len(total_df["group"].unique()) == 2:
        total_df["group"] = pd.Categorical(total_df["group"], categories=["low", "high"], ordered=True)
    else:
        total_df["group"] = pd.Categorical(total_df["group"], categories=["low", "mid", "high"], ordered=True)
    tot_dummies = pd.get_dummies(total_df["group"], prefix="grp", drop_first=True)
    tot_cox_data = pd.concat([total_df[["time", "event"]], tot_dummies], axis=1)

    try:
        tot_cph = CoxPHFitter()
        if tot_dummies.shape[1] > 0:
            formula = " + ".join(tot_dummies.columns)
            tot_cph.fit(tot_cox_data, duration_col="time", event_col="event", formula=formula)
            tot_hr_summary = tot_cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].to_dict(
                orient="index"
            )
        else:
            tot_cph.fit(tot_cox_data, duration_col="time", event_col="event")
            tot_hr_summary = {}
    except Exception:
        tot_hr_summary = {}

    tot_hr_rows = []
    for key, metrics in tot_hr_summary.items():
        comp = key.replace("grp_", "").capitalize()
        tot_hr_rows.append(
            html.Tr(
                [
                    html.Td(comp),
                    html.Td(f"{metrics['exp(coef)']:.2f}"),
                    html.Td(f"{metrics['exp(coef) lower 95%']:.2f}"),
                    html.Td(f"{metrics['exp(coef) upper 95%']:.2f}"),
                    html.Td(f"{metrics['p']:.4f}"),
                ]
            )
        )
    tot_hr_table = html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Comparison vs Low"),
                        html.Th("Hazard Ratio"),
                        html.Th("Lower 95% CI"),
                        html.Th("Upper 95% CI"),
                        html.Th("p-value"),
                    ]
                )
            ),
            html.Tbody(tot_hr_rows),
        ],
        style=table_style,
    )

    total_stats = html.Div(
        [
            html.H4("Total Cohort Statistics (using Optimal Cutpoints)"),
            html.Br(),
            tot_surv_table,
            html.Br(),
            tot_hr_table,
        ]
    )

    km_total_fig = km_figure(data, final_cut[0], final_cut[1])
    km_total_fig.update_layout(title="Total Cohort KM Curves (Optimal Cutpoints)")

    return cv_summary, hist_fig, cv_stats, total_stats, km_total_fig


if __name__ == "__main__":
    app.run(debug=True)
