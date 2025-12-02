import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#PAGE SETUP
st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")
st.title("Telco Customer Churn – Dashboard")


#LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    return df

df = load_data()


#SIDEBAR FILTERS
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


#TOP METRICS 
st.subheader("Overview")

col1, col2, col3, col4 = st.columns(4)

total_customers = len(filtered_df)
churn_rate = filtered_df["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
avg_monthly = filtered_df["MonthlyCharges"].mean()
avg_tenure = filtered_df["tenure"].mean()

col1.metric("Total Customers", total_customers)
col2.metric("Churn Rate", f"{churn_rate:.2f}%")
col3.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")
col4.metric("Avg Tenure", f"{avg_tenure:.1f} months")


#CHARTS ROW 1
st.markdown("---")
left, right = st.columns(2)

# 1) Churn distribution
with left:
    st.subheader("Churn Distribution")
    counts = filtered_df["Churn"].value_counts().reindex(["No", "Yes"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index, counts.values)
    ax.set_xlabel("Churn")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# 2) Churn by contract type (grouped bar)
with right:
    st.subheader("Churn by Contract Type")
    ct = pd.crosstab(filtered_df["Contract"], filtered_df["Churn"])
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(ct.index))
    width = 0.35

    ax.bar(x - width/2, ct["No"], width, label="No")
    ax.bar(x + width/2, ct["Yes"], width, label="Yes")

    ax.set_xticks(x)
    ax.set_xticklabels(ct.index, rotation=15)
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Count")
    ax.legend(title="Churn")
    st.pyplot(fig)


#CHARTS ROW 2 
st.markdown("---")
left2, right2 = st.columns(2)

# 3) Tenure distribution by churn (overlapping histograms)
with left2:
    st.subheader("Tenure Distribution by Churn")
    fig, ax = plt.subplots(figsize=(6, 4))
    tenure_no = filtered_df[filtered_df["Churn"] == "No"]["tenure"]
    tenure_yes = filtered_df[filtered_df["Churn"] == "Yes"]["tenure"]

    ax.hist(tenure_no, bins=30, alpha=0.6, label="No")
    ax.hist(tenure_yes, bins=30, alpha=0.6, label="Yes")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Count")
    ax.legend(title="Churn")
    st.pyplot(fig)

# 4) Monthly charges by churn (boxplot)
with right2:
    st.subheader("Monthly Charges by Churn")
    fig, ax = plt.subplots(figsize=(6, 4))
    data_to_plot = [
        filtered_df[filtered_df["Churn"] == "No"]["MonthlyCharges"],
        filtered_df[filtered_df["Churn"] == "Yes"]["MonthlyCharges"]
    ]
    ax.boxplot(data_to_plot, labels=["No", "Yes"])
    ax.set_xlabel("Churn")
    ax.set_ylabel("Monthly Charges ($)")
    st.pyplot(fig)


#CHARTS ROW 3
st.markdown("---")
left3, right3 = st.columns(2)

# 5) Churn by Internet Service Type
with left3:
    st.subheader("Churn by Internet Service Type")
    ct_int = pd.crosstab(filtered_df["InternetService"], filtered_df["Churn"])
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(ct_int.index))
    width = 0.35
    ax.bar(x - width/2, ct_int["No"], width, label="No")
    ax.bar(x + width/2, ct_int["Yes"], width, label="Yes")
    ax.set_xticks(x)
    ax.set_xticklabels(ct_int.index, rotation=15)
    ax.set_xlabel("Internet Service")
    ax.set_ylabel("Count")
    ax.legend(title="Churn")
    st.pyplot(fig)

# 6) Churn by Payment Method
with right3:
    st.subheader("Churn by Payment Method")
    ct_pay = pd.crosstab(filtered_df["PaymentMethod"], filtered_df["Churn"])
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(ct_pay.index))
    width = 0.35
    ax.bar(x - width/2, ct_pay["No"], width, label="No")
    ax.bar(x + width/2, ct_pay["Yes"], width, label="Yes")
    ax.set_xticks(x)
    ax.set_xticklabels(ct_pay.index, rotation=25, ha="right")
    ax.set_xlabel("Payment Method")
    ax.set_ylabel("Count")
    ax.legend(title="Churn")
    st.pyplot(fig)


#CORRELATION HEATMAP
st.markdown("---")
st.subheader("Correlation Heatmap (Numeric Features)")

numeric_df = filtered_df.select_dtypes(include="number")
corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(9, 7))
cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(np.arange(len(corr.columns)))
ax.set_yticks(np.arange(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(corr.columns)
fig.colorbar(cax)
st.pyplot(fig)


#RAW DATA
st.markdown("---")
with st.expander("Show Dataset"):
    st.dataframe(filtered_df)


