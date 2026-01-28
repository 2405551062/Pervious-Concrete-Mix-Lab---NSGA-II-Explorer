# Pervious Concrete Mix Lab (Python)

This is a minimal, runnable prototype of the interactive mix-design lab described in `ISC/index.txt`.

Quick start

1. Create and activate a Python environment (recommended Python 3.11+).

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

What is included

- `models.py` — simple empirical surrogates for strength, permeability and CO2.
- `optimizer.py` — NSGA-II wrapper using `pymoo` to search for Pareto-optimal mixes.
- `app.py` — Streamlit UI for running the optimizer and visualizing results.
- `requirements.txt` — Python dependencies.

Notes and next steps

- The surrogate models are illustrative. Replace with calibrated regressions or ML models for production.
- You can extract the `details` CSV for inspection and further analysis.
