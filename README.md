# Bhutan Healthcare Data Science Project

A Python-based data science workflow and Streamlit app for exploring Bhutan health indicators, building features, training models, and visualizing insights.

## Project Structure
- `data/raw/` Raw datasets (CSV, XLSX, etc.)
- `data/processed/` Outputs from notebooks and scripts (ignored by Git)
- `notebooks/`
  - `01_data_exploration.ipynb` Load and clean raw data → `cleaned.csv`
  - `02_feature_engineering.ipynb` Basic feature scaling → `features.csv`
  - `03_modeling.ipynb` Train RandomForest → `models/trained_model.pkl`
- `src/`
  - `data_loader.py` Load/validate/save processed datasets
  - `preprocessing.py` Minimal cleaning and encoding helpers
  - `train_model.py` CLI training script (classification/regression)
  - `scan_raw.py` Summarize all WHO datasets under `data/raw/`
- `app/`
  - `streamlit_app.py` Streamlit dashboard and prediction UI
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

## Training
- Train via notebook `03_modeling.ipynb` or CLI:
  - `python src/train_model.py --data data/processed/cleaned.csv --target stress_level --task classification --out models/trained_model.pkl`
- `starter.py` pipeline:
  - `python starter.py --raw "data/raw/<folder>/<Dataset_...>.csv" --target stress_level --task classification`

## Streamlit App
- Run locally: `streamlit run app/streamlit_app.py`
- Pages:
  - Raw Data Explorer: browse any dataset file, preview, dtypes, missingness, correlations
  - Dataset Overview: show processed/synthetic dataset, summary stats
  - Visualizations: heatmap, line, bar, area, histogram, scatter
  - Train Model Summary: feature importances when available
  - Predict Stress Level: sliders and prediction with trained/in-memory model
- If `data/processed/cleaned.csv` is absent, the app generates a synthetic dataset and trains in-memory to keep the UI functional.

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
