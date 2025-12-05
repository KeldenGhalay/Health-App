import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Bhutan Healthcare Analytics", layout="wide")
st.title("Bhutan Healthcare Data Science Project")

@st.cache_resource
def load_data():
    path = "data/processed/cleaned.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    np.random.seed(42)
    size = 600
    data = pd.DataFrame({
        "age": np.random.randint(15, 60, size),
        "sleep_hours": np.random.uniform(4, 10, size),
        "social_interaction": np.random.randint(0, 7, size),
        "work_stress": np.random.randint(1, 10, size),
        "physical_activity": np.random.randint(0, 6, size),
        "mood_score": np.random.randint(1, 10, size)
    })
    score = (data["work_stress"] * 0.5) + (10 - data["mood_score"]) + (6 - data["physical_activity"])
    conditions = [(score < 8), ((score >= 8) & (score < 14)), (score >= 14)]
    choices = ["low", "medium", "high"]
    data["stress_level"] = np.select(conditions, choices, default="medium")
    return data

@st.cache_resource
def load_or_train_model(data: pd.DataFrame):
    path = "models/trained_model.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    X = data.drop("stress_level", axis=1)
    y = data["stress_level"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

data = load_data()
model = load_or_train_model(data)

menu = st.sidebar.selectbox("Navigate", ["Raw Data Explorer", "Dataset Overview", "Visualizations", "Train Model Summary", "Predict Stress Level"])

if menu == "Raw Data Explorer":
    st.header("Raw Data Explorer")
    import glob, os
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
            st.subheader("Missingness by Column")
            miss = rdf.isna().mean().sort_values(ascending=False)
            st.bar_chart(miss)
            num = rdf.select_dtypes(include=["number"])
            if not num.empty:
                st.subheader("Correlation Heatmap")
                import seaborn as sns
                import matplotlib.pyplot as plt
                corr = num.corr()
                plt.figure(figsize=(8, 6))
                sns.heatmap(corr, annot=False, cmap="viridis")
                st.pyplot()
        except Exception as e:
            st.error(f"Failed to read {sel}: {e}")

elif menu == "Dataset Overview":
    st.header("Dataset Overview")
    st.dataframe(data.head())
    st.header("Summary of Data")
    st.dataframe(data.describe())
    if "stress_level" in data.columns:
        st.header("stress_level")
        st.bar_chart(data["stress_level"].value_counts())

elif menu == "Visualizations":
    st.header("Visualizations")
    viz_type = st.selectbox("Choose chart type:", ["Correlation Heatmap", "Line Chart", "Bar Chart", "Area Chart", "Histogram", "Scatter Plot"])
    if viz_type == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        numeric_data = data.select_dtypes(include=["number"])
        corr = numeric_data.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        st.pyplot()
    elif viz_type == "Line Chart":
        cols = data.columns.tolist()
        if "stress_level" in cols:
            st.line_chart(data.drop("stress_level", axis=1))
        else:
            st.line_chart(data)
    elif viz_type == "Bar Chart":
        feature = st.selectbox("Select feature:", data.columns[:-1])
        st.bar_chart(data[feature])
    elif viz_type == "Area Chart":
        cols = data.columns.tolist()
        if "stress_level" in cols:
            st.area_chart(data.drop("stress_level", axis=1))
        else:
            st.area_chart(data)
    elif viz_type == "Histogram":
        feature = st.selectbox("Select numeric feature:", data.select_dtypes(include=["number"]).columns)
        plt.hist(data[feature], bins=20)
        st.pyplot()
    elif viz_type == "Scatter Plot":
        numeric_cols = data.select_dtypes(include=["number"]).columns
        x_axis = st.selectbox("X-axis:", numeric_cols)
        y_axis = st.selectbox("Y-axis:", numeric_cols)
        plt.scatter(data[x_axis], data[y_axis])
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        st.pyplot()

elif menu == "Train Model Summary":
    st.header("Model Training Summary")
    if "stress_level" in data.columns:
        X = data.drop("stress_level", axis=1)
        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)
            st.bar_chart(importance_df.set_index("Feature"))
            st.dataframe(importance_df)
        else:
            st.write("Model does not expose feature importances.")
    else:
        st.write("Target column not found.")

elif menu == "Predict Stress Level":
    st.header("Predict Stress Level")
    if "stress_level" in data.columns:
        age = st.slider("Age", 15, 60, 25)
        sleep_hours = st.slider("Sleep Hours", 4.0, 10.0, 7.0)
        social_interaction = st.slider("Social Interaction", 0, 7, 3)
        work_stress = st.slider("Work/Study Stress", 1, 10, 5)
        physical_activity = st.slider("Physical Activity", 0, 6, 2)
        mood_score = st.slider("Mood Score", 1, 10, 6)
        if st.button("Predict"):
            features = np.array([[age, sleep_hours, social_interaction, work_stress, physical_activity, mood_score]])
            prediction = model.predict(features)[0]
            st.success(f"Predicted Stress Level: {prediction}")
    else:
        st.write("Target column not found.")
