# Bhutan Healthcare Data Science Project

A Python-based data science workflow and Streamlit app for exploring Bhutan health indicators, building features, training models, and visualizing insights.

## Project Structure
- `data/raw/` Raw datasets (CSV, XLSX, etc.)
- `data/processed/` Outputs from notebooks and scripts (ignored by Git)
- `notebooks/`
  - `01_data_exploration.ipynb` Load and clean raw data → `cleaned.csv`
  - `02_feature_engineering.ipynb` Build NCD trend features (YoY change, rolling mean)
  - `03_modeling.ipynb` Compute KPIs and visualize trends (no ML)
- `src/`
  - `data_loader.py` Load/validate/save processed datasets
  - `preprocessing.py` Minimal cleaning and encoding helpers
  - `train_model.py` CLI training script (classification/regression)
  - `scan_raw.py` Summarize all WHO datasets under `data/raw/`
- `app/`
  - `streamlit_app.py` Streamlit dashboard focused on NCD premature mortality
- `models/` Saved model artifacts (ignored by Git)
- `starter.py` Simple pipeline runner from a single raw CSV
- `requirements.txt` Python dependencies
- `.gitignore` Excludes caches, virtual env, processed outputs, models

## Setup
- Python 3.10+
- Create and activate venv
  - Windows: `python -m venv .venv && .venv\Scripts\activate`
  - macOS/Linux: `python -m venv venv && source venv/bin/activate`
- Install deps: `pip install -r requirements.txt`

## Data
- Place Bhutan health datasets under `data/raw/` (WHO GHO, etc.).
- Summarize all raw datasets: `python src/scan_raw.py` → `data/processed/raw_summary.csv`.
- Notebooks generate `data/processed/cleaned.csv` and `features.csv`.

## Notebooks
- Open each notebook and run cells in order.
- Outputs are written to `data/processed/`.

## Optional Training (not required)
- Generic CLI (regression on NCD rate):
  - `python src/train_model.py --data data/processed/cleaned.csv --target RATE_PER_100_N --task regression --out models/trained_model.pkl`
- `starter.py` pipeline:
  - `python starter.py --raw "data/raw/<folder>/<Dataset_...>.csv"`

## Streamlit App
- Run locally: `streamlit run app/streamlit_app.py`
- Pages:
  - Raw Data Explorer: browse any dataset file, preview, dtypes, missingness, correlations
  - NCD Overview: preview, summary, years and sex categories, average rate
  - Trends: line chart of `RATE_PER_100_N` by year and sex, change KPI
  - Sex Comparison: average rates by sex; per-year area chart
  - Uncertainty Bands: ribbon chart using `RATE_PER_100_NL` and `RATE_PER_100_NU`
  - Train Model Summary: trains a synthetic NCD risk classifier and shows feature importances
  - Predict NCD: input factors (age, sex, smoking, physical activity, BMI, BP, glucose, cholesterol, family history) → prediction and probability

## Deploy to Streamlit Cloud
1) Push to GitHub:
- `git init && git add . && git commit -m "Initial commit"`
- `git branch -M main`
- `git remote add origin https://github.com/<user>/<repo>.git`
- `git push -u origin main`
2) On `https://streamlit.io/cloud`:
- New app → connect your repo
- Branch: `main`
- File: `app/streamlit_app.py`
- Confirm `requirements.txt` is detected → Deploy

## Notes
- `data/processed/` and `models/` are ignored by Git. Commit a small `cleaned.csv` if you want real data on the cloud; otherwise the app uses synthetic data.
- Customize `preprocessing.py` for indicator-specific cleaning and imputation.
- Enhance `streamlit_app.py` with thematic dashboards (WASH, maternal/child, communicable/NCD).

## License
- Add a license of your choice (MIT recommended).
