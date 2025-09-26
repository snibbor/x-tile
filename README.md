# x-tile
![image](https://github.com/user-attachments/assets/08c9188e-5c6b-4c1d-9fa8-ad04e5533ddc)


X-Tile is a graphical method for biomarker assessment and outcome-based cut-point optimization, originally described by Camp, Dolled-Filhart, and Rimm in 2004, which constructs a two-dimensional projection of every possible low/high threshold pair to identify statistically robust divisions of patient cohorts based on survival outcomes  ([X-tile: a new bio-informatics tool for biomarker assessment and ...](https://pubmed.ncbi.nlm.nih.gov/15534099/?utm_source=chatgpt.com)). This repository, **x-tile**, provides a modern Python implementation of that methodology, featuring core functions for grouping and log-rank statistic computation, automated candidate grid generation over percentiles of biomarker distributions, and interactive visualizations via a Dash web application  ([x-tile/x_tile_app.py at main · snibbor/x-tile · GitHub](https://github.com/snibbor/x-tile/blob/main/x_tile_app.py), [x-tile/x_tile_app.py at main · snibbor/x-tile · GitHub](https://github.com/snibbor/x-tile/blob/main/x_tile_app.py)). An accompanying Jupyter notebook demonstrates simulated data workflows, heatmap visualization, and Kaplan–Meier curve plotting, enabling users to reproduce X-Tile analyses programmatically or through a user-friendly interface  ([github.com](https://github.com/snibbor/x-tile/raw/refs/heads/main/x-tile.ipynb)).

## Installation

To install **x-tile**, you need Python 3.8 or later. Clone the repository and install dependencies:

```bash
git clone https://github.com/snibbor/x-tile.git
cd x-tile
pip install -r requirements.txt
```
 ([x-tile/x_tile_app.py at main · snibbor/x-tile · GitHub](https://github.com/snibbor/x-tile/blob/main/x_tile_app.py)) Dependencies include Dash and Dash Bootstrap Components for the web UI, Plotly for plotting, Pandas and NumPy for data handling, Lifelines for survival analysis, scikit-learn for data splitting, and Diskcache for background callbacks  ([x-tile/x_tile_app.py at main · snibbor/x-tile · GitHub](https://github.com/snibbor/x-tile/blob/main/x_tile_app.py)).

## Usage

### Interactive Dash App

Run the Dash application:

```bash
python x_tile_app.py
```

Then open [http://127.0.0.1:8050](http://127.0.0.1:8050) in your browser to:

- **Upload** a CSV or Excel file containing columns for biomarker, follow-up time, and event indicator.  
- **Map** your columns if they aren’t named exactly `biomarker`, `time`, and `event`.  
- **Explore** the X-Tile heatmap, select cutpoints via clicking, and view Kaplan–Meier and histogram plots  ([x-tile/x_tile_app.py at main · snibbor/x-tile · GitHub](https://github.com/snibbor/x-tile/blob/main/x_tile_app.py)).

### As a Python Library

You can import core functions directly:

```python
from x_tile_app import (
    build_candidate_grid,
    make_heatmap_figure,
    km_figure,
    hist_figure,
    cross_validate_cutpoints
)

# Example:
low_vals, high_vals, matrix = build_candidate_grid(your_dataframe)
fig = make_heatmap_figure(low_vals, high_vals, matrix)
fig.show()
```
 ([x-tile/x_tile_app.py at main · snibbor/x-tile · GitHub](https://github.com/snibbor/x-tile/blob/main/x_tile_app.py))

### Jupyter Notebook Demonstration

The `x-tile.ipynb` notebook walks through:

1. **Simulating** biomarker, survival times, and censoring.  
2. **Computing** optimal low/high cutpoints by maximizing the chi-square from a multivariate log-rank test.  
3. **Plotting** the X-Tile heatmap, Kaplan–Meier curves, and biomarker histograms.  

It replicates the original X-Tile workflow and provides ready-to-run example code  ([github.com](https://github.com/snibbor/x-tile/raw/refs/heads/main/x-tile.ipynb)).

## Features

### Data Handling & Helper Functions

- **assign_groups**: Splits continuous biomarker values into “low”, “mid”, and “high” based on two thresholds.  
- **compute_logrank_statistic**: Calculates the chi-square statistic from a multivariate log-rank test for survival differences among groups.  
- **compute_direction**: Determines whether higher biomarker values associate with better or worse survival.  
These functions leverage the Lifelines library for rigorous survival analysis  ([x-tile/x_tile_app.py at main · snibbor/x-tile · GitHub](https://github.com/snibbor/x-tile/blob/main/x_tile_app.py)).

### Candidate Grid Generation

- **build_candidate_grid**: Generates cutpoint candidates at the 5th–95th percentile of biomarker values, computes signed chi-square statistics for all valid low ≤ high pairs, and outputs arrays of low cuts, high cuts, and the resulting matrix  ([x-tile/x_tile_app.py at main · snibbor/x-tile · GitHub](https://github.com/snibbor/x-tile/blob/main/x_tile_app.py)).

### Visualization Helpers

- **make_heatmap_figure**: Creates an X-Tile heatmap with a diverging “RdYlGn” colorscale to highlight optimal regions.  
- **km_figure**: Generates Kaplan–Meier survival curves for each biomarker-defined group.  
- **hist_figure**: Plots biomarker distributions with vertical lines at selected cutpoints.  
These are built using Plotly for interactive, publication-quality figures  ([x-tile/x_tile_app.py at main · snibbor/x-tile · GitHub](https://github.com/snibbor/x-tile/blob/main/x_tile_app.py), [x-tile/x_tile_app.py at main · snibbor/x-tile · GitHub](https://github.com/snibbor/x-tile/blob/main/x_tile_app.py)).

### Monte Carlo Cross-Validation

- **cross_validate_cutpoints**: Runs repeated (parallelized) Monte Carlo cross-validation to test cutpoint stability, aggregates median cutpoints, p-values, median survivals, and hazard ratios across iterations, and provides summary statistics and plots  ([x-tile/x_tile_app.py at main · snibbor/x-tile · GitHub](https://github.com/snibbor/x-tile/blob/main/x_tile_app.py)).

## Citation

If you use **x-tile** in your research, please cite this repository and the original paper:
- Robbins CJ. snibbor/x‑tile: Modern Python implementation of X‑Tile (GitHub repository). 2025. https://github.com/snibbor/x-tile

- Camp RL, Dolled-Filhart M, Rimm DL. X-Tile: a new bio-informatics tool for biomarker assessment and outcome-based cut-point optimization. *Clin Cancer Res.* 2004 Nov 1;10(21):7252–9. doi:10.1158/1078-0432.CCR-04-0713  ([X-tile: a new bio-informatics tool for biomarker assessment and ...](https://pubmed.ncbi.nlm.nih.gov/15534099/)).

## Acknowledgments

This implementation modernizes the X-Tile methodology first introduced by Camp et al. at Yale University  ([X-tile: a new bio-informatics tool for biomarker assessment and ...](https://pubmed.ncbi.nlm.nih.gov/15534099/)). Contributions and feedback to this repository are welcome via GitHub issues.

