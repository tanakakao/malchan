# HTML report visualization parity

The detailed HTML report collects figures from the existing FastAPI visualization and XAI APIs at download time, converts Plotly figures to embedded PNG data URLs, and keeps the downloaded report self-contained.

Included legacy Excel export equivalents:

- regression Y–Y and residual plots, including cross-validation train/validation layers when available
- classification confusion matrices
- normalized model, SHAP, and permutation feature importance comparison
- class/output-specific SHAP beeswarm plots
- one-dimensional partial dependence with baseline + SHAP, observed values, and baseline overlays
- two-dimensional partial dependence for numeric feature pairs

Figure collection is fault tolerant. A missing XAI result is rendered as an explanatory card in the HTML instead of aborting the complete report download.
