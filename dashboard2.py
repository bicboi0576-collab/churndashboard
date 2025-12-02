import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")
st.title("Telco Customer Churn – Interactive Dashboard")


# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    # Make sure this CSV file is in the same folder as this script
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    return df


df = load_data()


# -------------------- SIDEBAR FILTERS --------------------
st.sidebar.header("Filters")

# Contract filter
contract_list = df["Contract"].unique().tolist()
contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=contract_list,
    default=contract_list
)

# Churn filter
churn_list = df["Churn"].unique().tolist()
churn_filter = st.sidebar.multiselect(
    "Churn Status",
    options=churn_list,
    default=churn_list
)

# Internet service filter
internet_list = df["InternetService"].unique().tolist()
internet_filter = st.sidebar.multiselect(
    "Internet Service Type",
    options=internet_list,
    default=internet_list
)

# Tenure slider
tenure_min = int(df["tenure"].min())
tenure_max = int(df["tenure"].max())
tenure_filter = st.sidebar.slider(
    "Tenure Range (months)",
    min_value=tenure_min,
    max_value=tenure_max,
    value=(tenure_min, tenure_max)
)

# Apply filters
filtered_df = df[
    (df["Contract"].isin(contract_filter)) &
    (df["Churn"].isin(churn_filter)) &
    (df["InternetService"].isin(internet_filter)) &
    (df["tenure"].between(tenure_filter[0], tenure_filter[1]))
]

if filtered_df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()


# -------------------- TOP METRICS --------------------
st.subheader("Overview (After Filters)")

col1, col2, col3, col4 = st.columns(4)

total_customers = len(filtered_df)
churn_rate = filtered_df["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
avg_monthly = filtered_df["MonthlyCharges"].mean()
avg_tenure = filtered_df["tenure"].mean()

col1.metric("Total Customers", total_customers)
col2.metric("Churn Rate", f"{churn_rate:.2f}%")
col3.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")
col4.metric("Avg Tenure", f"{avg_tenure:.1f} months")


# -------------------- CHARTS ROW 1 --------------------
st.markdown("---")
left, right = st.columns(2)

# Churn distribution (hover shows count + label)
with left:
    st.subheader("Churn Distribution")
    fig = px.histogram(
        filtered_df,
        x="Churn",
        color="Churn",
        text_auto=True,
        title="Churn Count"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Churn by contract type (hover shows contract, churn, count)
with right:
    st.subheader("Churn by Contract Type")
    fig = px.histogram(
        filtered_df,
        x="Contract",
        color="Churn",
        barmode="group",
        text_auto=True,
        title="Churn by Contract Type"
    )
    fig.update_layout(xaxis_title="Contract", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)


# -------------------- CHARTS ROW 2 --------------------
st.markdown("---")
left2, right2 = st.columns(2)

# Tenure distribution (hover shows tenure, count, churn)
with left2:
    st.subheader("Tenure Distribution by Churn")
    fig = px.histogram(
        filtered_df,
        x="tenure",
        color="Churn",
        nbins=40,
        marginal="box",
        title="Tenure Distribution"
    )
    fig.update_layout(xaxis_title="Tenure (months)", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

# Monthly charges by churn (hover shows churn + charge value)
with right2:
    st.subheader("Monthly Charges by Churn Group")
    fig = px.box(
        filtered_df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Monthly Charges by Churn"
    )
    fig.update_layout(xaxis_title="Churn", yaxis_title="Monthly Charges ($)")
    st.plotly_chart(fig, use_container_width=True)


# -------------------- RAW DATA --------------------
st.markdown("---")
with st.expander("Show Raw Filtered Data"):
    st.dataframe(filtered_df)

