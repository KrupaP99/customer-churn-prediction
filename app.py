"""
app.py
Interactive Streamlit dashboard for the churn prediction project.
Run with: streamlit run app.py
"""

import sys
sys.path.append('src')

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from load_data import load_raw_data
from clean_data import clean_data

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

# ---- Load and cache data (so it doesn't reload on every interaction) ----
@st.cache_data
def get_data():
    raw_df = load_raw_data()
    df = clean_data(raw_df)
    return df

df = get_data()

st.title("📊 Customer Churn & Revenue Analytics")
st.caption("IBM Telco Customer Churn dataset — 7,043 customers")

# ---- Sidebar filter (this is the interactive part) ----
st.sidebar.header("Filters")
contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=df["Contract"].unique(),
    default=df["Contract"].unique()
)

filtered_df = df[df["Contract"].isin(contract_filter)]

# ---- Headline metrics ----
col1, col2, col3 = st.columns(3)

churn_rate = filtered_df["Churn"].mean()
avg_arpu = filtered_df["MonthlyCharges"].mean()
revenue_at_risk = filtered_df[filtered_df["Churn"] == 1]["MonthlyCharges"].sum()

col1.metric("Churn Rate", f"{churn_rate:.1%}")
col2.metric("Avg Monthly Charge (ARPU)", f"${avg_arpu:.2f}")
col3.metric("Monthly Revenue at Risk", f"${revenue_at_risk:,.0f}")

st.divider()

# ---- Charts ----
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Rate by Contract Type")
    contract_churn = filtered_df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    contract_churn.plot(kind="bar", color="#C44E52", ax=ax)
    ax.set_ylabel("Churn Rate")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    st.pyplot(fig)

with col2:
    st.subheader("Churn Rate by Payment Method")
    payment_churn = filtered_df.groupby("PaymentMethod")["Churn"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    payment_churn.plot(kind="barh", color="#8172B2", ax=ax)
    ax.set_xlabel("Churn Rate")
    st.pyplot(fig)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Average Monthly Charges: Churned vs Retained")
    arpu_by_churn = filtered_df.groupby("Churn")["MonthlyCharges"].mean()
    fig, ax = plt.subplots()
    arpu_by_churn.plot(kind="bar", color=["#4C72B0", "#DD8452"], ax=ax)
    ax.set_xticklabels(["Stayed", "Churned"], rotation=0)
    ax.set_ylabel("Avg Monthly Charges ($)")
    st.pyplot(fig)

with col4:
    st.subheader("Churn Rate by Tenure Group")
    tenure_df = filtered_df.copy()
    tenure_df["tenure_group"] = pd.cut(
        tenure_df["tenure"], bins=[0, 12, 24, 48, 72],
        labels=["0-1 yr", "1-2 yr", "2-4 yr", "4-6 yr"]
    )
    tenure_churn = tenure_df.groupby("tenure_group", observed=True)["Churn"].mean()
    fig, ax = plt.subplots()
    tenure_churn.plot(kind="bar", color="#55A868", ax=ax)
    ax.set_ylabel("Churn Rate")
    plt.xticks(rotation=0)
    st.pyplot(fig)

st.divider()
st.caption("Filter by contract type in the sidebar to see how the metrics change.")