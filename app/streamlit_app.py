import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import glob

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

data = load_ncd_data()
menu = st.sidebar.selectbox("Navigate", ["Raw Data Explorer", "NCD Overview", "Trends", "Sex Comparison", "Uncertainty Bands"]) 

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
            st.subheader("Missingness by Column")
            miss = rdf.isna().mean().sort_values(ascending=False)
            st.bar_chart(miss)
            num = rdf.select_dtypes(include=["number"])
            if not num.empty:
                st.subheader("Correlation Heatmap")
                corr = num.corr()
                plt.figure(figsize=(8, 6))
                sns.heatmap(corr, annot=False, cmap="viridis")
                st.pyplot()
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
