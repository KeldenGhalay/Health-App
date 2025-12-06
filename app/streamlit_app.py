import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import joblib
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Bhutan NCD Premature Mortality (30–70)", layout="wide")
st.title("Bhutan NCD Premature Mortality (Ages 30–70)")

@st.cache_resource
def find_ncd_dataset():
    candidates = glob.glob(os.path.join("data", "raw", "**", "*1F96863*Dataset_*.csv"), recursive=True)
    if candidates:
        return candidates[0]
    # fallback: any dataset file
    any_ds = glob.glob(os.path.join("data", "raw", "**", "*Dataset_*.csv"), recursive=True)
    return any_ds[0] if any_ds else None

@st.cache_resource
def load_ncd_data(path: str | None = None) -> pd.DataFrame | None:
    if path and os.path.exists(path):
        return pd.read_csv(path)
    auto = find_ncd_dataset()
    if auto:
        return pd.read_csv(auto)
    return None

@st.cache_resource
def generate_ncd_risk_data(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    age = rng.integers(30, 71, n)
    sex = rng.choice(["Male","Female"], n)
    smoking = rng.integers(0, 2, n)
    physical_activity = rng.integers(0, 7, n)
    bmi = rng.uniform(18.0, 40.0, n)
    systolic_bp = rng.uniform(100.0, 180.0, n)
    fasting_glucose = rng.uniform(80.0, 200.0, n)
    cholesterol = rng.uniform(150.0, 300.0, n)
    family_history = rng.integers(0, 2, n)
    sex_male = (sex == "Male").astype(int)
    score = (
        0.03 * (age - 30)
        + 0.8 * smoking
        - 0.2 * physical_activity
        + 0.05 * (bmi - 25)
        + 0.02 * (systolic_bp - 120)
        + 0.02 * (fasting_glucose - 100)
        + 0.01 * (cholesterol - 200)
        + 0.7 * family_history
        + 0.2 * sex_male
    )
    p = 1.0 / (1.0 + np.exp(-score / 3.0))
    has_ncd = np.where(p >= 0.5, "Yes", "No")
    return pd.DataFrame({
        "age": age,
        "sex": sex,
        "smoking": smoking,
        "physical_activity": physical_activity,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "fasting_glucose": fasting_glucose,
        "cholesterol": cholesterol,
        "family_history": family_history,
        "has_ncd": has_ncd,
    })

data = load_ncd_data()
menu = st.sidebar.selectbox("Navigate", ["Raw Data Explorer", "NCD Overview", "Trends", "Sex Comparison", "Uncertainty Bands", "Train Model Summary", "Predict NCD"]) 

if menu == "Raw Data Explorer":
    st.header("Raw Data Explorer")
    dataset_files = glob.glob(os.path.join("data", "raw", "**", "*Dataset_*.csv"), recursive=True)
    if not dataset_files:
        st.info("No WHO dataset CSVs found in data/raw.")
    else:
        sel = st.selectbox("Select dataset file", dataset_files)
        try:
            rdf = pd.read_csv(sel)
            st.write(f"Shape: {rdf.shape}")
            st.dataframe(rdf.head())
            dtypes_df = pd.DataFrame({"Column": rdf.columns, "Dtype": rdf.dtypes.astype(str)})
            st.subheader("Column Types")
            st.dataframe(dtypes_df)
        except Exception as e:
            st.error(f"Failed to read {sel}: {e}")

elif menu == "NCD Overview":
    st.header("NCD Premature Mortality Overview")
    if data is None:
        st.warning("No dataset found. Place the NCD dataset under data/raw.")
    else:
        st.write("Preview")
        st.dataframe(data.head())
        st.write("Summary")
        st.dataframe(data.describe(include="all"))
        if {"DIM_TIME", "Sex", "RATE_PER_100_N"}.issubset(set(data.columns)):
            years = sorted(data["DIM_TIME"].unique())
            sexes = sorted(data["Sex"].unique())
            st.write(f"Years: {years[0]}–{years[-1]} ({len(years)} values)")
            st.write(f"Sex categories: {', '.join(sexes)}")
            overall_mean = data["RATE_PER_100_N"].mean()
            st.metric("Average premature NCD mortality (per 100, ages 30–70)", f"{overall_mean:.1f}")

elif menu == "Trends":
    st.header("Trends Over Time")
    if data is None or not {"DIM_TIME", "Sex", "RATE_PER_100_N"}.issubset(set(data.columns)):
        st.warning("Dataset missing required columns.")
    else:
        pivot = data.pivot_table(index="DIM_TIME", columns="Sex", values="RATE_PER_100_N")
        st.line_chart(pivot)
        first_year = pivot.index.min()
        last_year = pivot.index.max()
        if "Total" in pivot.columns:
            delta = pivot.loc[last_year, "Total"] - pivot.loc[first_year, "Total"]
            st.metric("Change in total rate (last vs first year)", f"{delta:+.1f}")
        else:
            for sex in pivot.columns:
                delta = pivot.loc[last_year, sex] - pivot.loc[first_year, sex]
                st.metric(f"Change ({sex})", f"{delta:+.1f}")

elif menu == "Sex Comparison":
    st.header("Sex Comparison")
    if data is None or not {"Sex", "RATE_PER_100_N", "DIM_TIME"}.issubset(set(data.columns)):
        st.warning("Dataset missing required columns.")
    else:
        avg = data.groupby("Sex")["RATE_PER_100_N"].mean().sort_values(ascending=False)
        st.bar_chart(avg)
        st.write("Rates by year and sex")
        pivot = data.pivot_table(index="DIM_TIME", columns="Sex", values="RATE_PER_100_N")
        st.area_chart(pivot)

elif menu == "Uncertainty Bands":
    st.header("Uncertainty Bands")
    if data is None or not {"DIM_TIME", "Sex", "RATE_PER_100_N", "RATE_PER_100_NL", "RATE_PER_100_NU"}.issubset(set(data.columns)):
        st.warning("Dataset missing required columns.")
    else:
        sex = st.selectbox("Select sex", sorted(data["Sex"].unique()))
        d = data[data["Sex"] == sex].sort_values("DIM_TIME")
        plt.figure(figsize=(8, 5))
        plt.plot(d["DIM_TIME"], d["RATE_PER_100_N"], label="Estimate", color="#1f77b4")
        plt.fill_between(d["DIM_TIME"], d["RATE_PER_100_NL"], d["RATE_PER_100_NU"], color="#1f77b4", alpha=0.2, label="Uncertainty")
        plt.xlabel("Year")
        plt.ylabel("Rate per 100 (ages 30–70)")
        plt.legend()
        st.pyplot()

elif menu == "Train Model Summary":
    st.header("Train Model Summary")
    df = generate_ncd_risk_data(1200)
    X = df[["age","smoking","physical_activity","bmi","systolic_bp","fasting_glucose","cholesterol","family_history","sex"]]
    X = pd.get_dummies(X, columns=["sex"], drop_first=True)
    y = df["has_ncd"]
    model = RandomForestClassifier()
    model.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "features": list(X.columns)}, "models/trained_ncd_model.pkl")
    imps = pd.DataFrame({"Feature": list(X.columns), "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)
    st.bar_chart(imps.set_index("Feature"))
    st.dataframe(imps)
    st.write("Factors used: age, sex, smoking, physical_activity, bmi, systolic_bp, fasting_glucose, cholesterol, family_history")

elif menu == "Predict NCD":
    st.header("Predict NCD")
    model_path = "models/trained_ncd_model.pkl"
    if os.path.exists(model_path):
        obj = joblib.load(model_path)
        if hasattr(obj, "predict_proba"):
            model = obj
            feature_names = list(getattr(model, "feature_names_in_", []))
        elif isinstance(obj, dict) and "model" in obj:
            model = obj["model"]
            feature_names = obj.get("features", list(getattr(model, "feature_names_in_", [])))
        else:
            model = obj
            feature_names = list(getattr(model, "feature_names_in_", []))
    else:
        df = generate_ncd_risk_data(1200)
        X = df[["age","smoking","physical_activity","bmi","systolic_bp","fasting_glucose","cholesterol","family_history","sex"]]
        X = pd.get_dummies(X, columns=["sex"], drop_first=True)
        y = df["has_ncd"]
        model = RandomForestClassifier()
        model.fit(X, y)
        feature_names = list(X.columns)
    age = st.slider("Age", 30, 70, 45)
    sex = st.selectbox("Sex", ["Male","Female"])
    smoking = st.selectbox("Smoking", ["No","Yes"]) == "Yes"
    physical_activity = st.slider("Physical Activity (days/wk)", 0, 6, 2)
    bmi = st.slider("BMI", 18.0, 40.0, 26.0)
    systolic_bp = st.slider("Systolic BP", 90.0, 180.0, 125.0)
    fasting_glucose = st.slider("Fasting Glucose", 70.0, 200.0, 105.0)
    cholesterol = st.slider("Cholesterol", 150.0, 300.0, 210.0)
    family_history = st.selectbox("Family History", ["No","Yes"]) == "Yes"
    if st.button("Predict"):
        row = {
            "age": age,
            "smoking": 1 if smoking else 0,
            "physical_activity": physical_activity,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "fasting_glucose": fasting_glucose,
            "cholesterol": cholesterol,
            "family_history": 1 if family_history else 0,
            "sex": sex,
        }
        X = pd.DataFrame([row])
        if "sex" in X.columns:
            X["sex_Male"] = 1 if sex == "Male" else 0
            X = X.drop(columns=["sex"])
        train_feature_names = list(getattr(model, "feature_names_in_", []))
        if not train_feature_names:
            train_feature_names = feature_names or list(X.columns)
        for f in train_feature_names:
            if f not in X.columns:
                X[f] = 0
        X = X[train_feature_names]
        X = X.astype(float)
        proba = model.predict_proba(X)
        classes = list(model.classes_)
        idx = classes.index("Yes") if "Yes" in classes else 1
        p_yes = float(proba[0][idx])
        pred = "Yes" if p_yes >= 0.5 else "No"
        st.metric("Predicted NCD", pred)
        st.metric("Probability (Yes)", f"{p_yes:.2f}")
