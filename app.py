import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent

def main():
    st.set_page_config(layout="wide", page_title="CO₂ Emissions Dashboard")

    @st.cache_data
    def load_data():
        return pd.read_csv(BASE_DIR / "emissions.csv")

    @st.cache_resource
    def load_model():
        mdl = joblib.load(BASE_DIR / "best_model.pkl")
        scl = joblib.load(BASE_DIR / "scaler.pkl")
        return mdl, scl

    df = load_data()
    model, scaler = load_model()

    tabs = st.tabs(["Overview", "EDA", "Predictor"])

    # --- Overview tab ---
    with tabs[0]:
        latest_year = df.year.max()
        total_latest = df.loc[df.year == latest_year, 'value'].sum()
        total_prev   = df.loc[df.year == latest_year - 1, 'value'].sum()
        yoy_change   = (total_latest - total_prev) / total_prev if total_prev else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Year", latest_year)
        col2.metric("Total Emissions", f"{total_latest:,.0f} Mt CO₂", delta=f"{yoy_change:.1%}")
        highest_state = (
            df[df.year == latest_year]
            .groupby('state-name')['value'].sum()
            .idxmax()
        )
        col3.metric("Highest State", highest_state)

    # --- EDA tab ---
    with tabs[1]:
        sector_totals = df.groupby(['year', 'sector-name'])['value'].sum().unstack()
        st.line_chart(sector_totals)

    # --- Predictor tab ---
    with tabs[2]:
        with st.form("pred_form"):
            year   = st.slider(
                "Year",
                int(df.year.min()),
                int(df.year.max()),
                int(df.year.max())
            )
            state  = st.selectbox("State", sorted(df['state-name'].unique()))
            fuel   = st.selectbox("Fuel", sorted(df['fuel-name'].unique()))
            sector = st.selectbox("Sector", sorted(df['sector-name'].unique()))
            submitted = st.form_submit_button("Predict")

        if submitted:
            # build all 6 features
            year_norm = year - int(df.year.min())
            fuels = df['fuel-name'].unique().tolist()
            sectors = df['sector-name'].unique().tolist()
            fuel_num = fuels.index(fuel)
            sector_num = sectors.index(sector)

            subset = df.query("year == @year and `state-name` == @state")
            total = subset['value'].sum()
            coal_share = (
                subset.loc[subset['fuel-name'] == 'Coal', 'value'].sum() / total
                if total else 0
            )
            gas_share = (
                subset.loc[subset['fuel-name'] == 'Natural Gas', 'value'].sum() / total
                if total else 0
            )
            sector_fuel_int = sector_num * fuel_num

            feats = np.array([[
                year_norm,
                fuel_num,
                sector_num,
                coal_share,
                gas_share,
                sector_fuel_int
            ]])

            # scale all 6, then pick only the first model.n_features_in_ columns
            scaled6 = scaler.transform(feats)
            scaled_for_model = scaled6[:, : model.n_features_in_]

            pred = model.predict(scaled_for_model)[0]
            prob = model.predict_proba(scaled_for_model)[0][1]

            st.metric(
                "High-emitter?",
                "Yes" if pred else "No",
                delta=f"{prob:.1%}"
            )

if __name__ == "__main__":
    main()
