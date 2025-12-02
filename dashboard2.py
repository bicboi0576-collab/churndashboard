import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")
st.title("📊 Telco Customer Churn – Dashboard (No Hover Version)")


# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
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
st.subheader("📌 Overview (After Filters)")

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

# Churn distribution
with left:
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(data=filtered_df, x="Churn", ax=ax)
    st.pyplot(fig)

# Churn by contract type
with right:
    st.subheader("Churn by Contract Type")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(data=filtered_df, x="Contract", hue="Churn", ax=ax)
    plt.xticks(rotation=15)
    st.pyplot(fig)


# -------------------- CHARTS ROW 2 --------------------
st.markdown("---")
left2, right2 = st.columns(2)

# Tenure distribution
with left2:
    st.subheader("Tenure Distribution by Churn")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.kdeplot(data=filtered_df, x="tenure", hue="Churn", fill=True, ax=ax)
    st.pyplot(fig)

# Monthly charges by churn
with right2:
    st.subheader("Monthly Charges by Churn")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(data=filtered_df, x="Churn", y="MonthlyCharges", ax=ax)
    st.pyplot(fig)


# -------------------- CHARTS ROW 3 (ADDED) --------------------
st.markdown("---")
left3, right3 = st.columns(2)

# Churn by Internet Service
with left3:
    st.subheader("Churn by Internet Service Type")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(data=filtered_df, x="InternetService", hue="Churn", ax=ax)
    plt.xticks(rotation=15)
    st.pyplot(fig)

# Churn by Payment Method
with right3:
    st.subheader("Churn by Payment Method")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(data=filtered_df, x="PaymentMethod", hue="Churn", ax=ax)
    plt.xticks(rotation=25)
    st.pyplot(fig)


# -------------------- CORRELATION HEATMAP --------------------
st.markdown("---")
st.subheader("Correlation Heatmap (Numeric Features)")

numeric_df = filtered_df.select_dtypes(include="number")
corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
st.pyplot(fig)


# -------------------- RAW DATA --------------------
st.markdown("---")
with st.expander("Show Raw Filtered Data"):
    st.dataframe(filtered_df)
