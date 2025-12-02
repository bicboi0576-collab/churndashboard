import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

sns.set(style="whitegrid")

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")
st.title("Telco Customer Churn Dashboard")

# ------------------ OVERVIEW TEXT ------------------ #
st.markdown("""
### Overview
""")


# ------------------ LOAD & PREP DATA ------------------ #
@st.cache_data
def load_and_prepare():
    # Load raw data
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Clean TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    # Copy with original categorical labels (for filters & some charts)
    df_raw = df.copy()

    # Model/correlation version: drop ID and encode categoricals
    df_model = df_raw.drop(columns=["customerID"])
    le_dict = {}
    for col in df_model.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        le_dict[col] = le

    # Train/test split for model (same as notebook)
    X = df_model.drop("Churn", axis=1)
    y = df_model["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top15 = importances.sort_values(ascending=False).head(15)

    return df_raw, df_model, cm, top15


df_raw, df_model, cm, top15 = load_and_prepare()


# ------------------ SIDEBAR FILTERS ------------------ #
st.sidebar.header("Filters")

contract_options = df_raw["Contract"].unique().tolist()
contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=contract_options,
    default=contract_options
)

churn_options = df_raw["Churn"].unique().tolist()
churn_filter = st.sidebar.multiselect(
    "Churn Status",
    options=churn_options,
    default=churn_options
)

internet_options = df_raw["InternetService"].unique().tolist()
internet_filter = st.sidebar.multiselect(
    "Internet Service Type",
    options=internet_options,
    default=internet_options
)

tenure_min = int(df_raw["tenure"].min())
tenure_max = int(df_raw["tenure"].max())
tenure_filter = st.sidebar.slider(
    "Tenure Range (months)",
    min_value=tenure_min,
    max_value=tenure_max,
    value=(tenure_min, tenure_max)
)

# Apply filters to df_raw (these affect metrics and some charts)
filtered_df = df_raw[
    (df_raw["Contract"].isin(contract_filter)) &
    (df_raw["Churn"].isin(churn_filter)) &
    (df_raw["InternetService"].isin(internet_filter)) &
    (df_raw["tenure"].between(tenure_filter[0], tenure_filter[1]))
]

if filtered_df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()


# ------------------ OVERVIEW METRICS ------------------ #
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


# ------------------ ROW 1: CHURN COUNT & CONTRACT ------------------ #
st.markdown("---")
row1_col1, row1_col2 = st.columns(2)

# 1) Churn vs Non-Churn Count (uses filtered data)
with row1_col1:
    st.subheader("Churn vs Non-Churn Count")
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    sns.countplot(data=filtered_df, x="Churn", ax=ax1)
    ax1.set_title("Churn vs Non-Churn Count")
    ax1.set_xlabel("Churn")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

# 2) Churn Rate by Contract Type (uses filtered data)
with row1_col2:
    st.subheader("Churn Rate by Contract Type")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.countplot(data=filtered_df, x="Contract", hue="Churn", ax=ax2)
    ax2.set_title("Churn Rate by Contract Type")
    ax2.set_xlabel("Contract")
    ax2.set_ylabel("count")
    ax2.tick_params(axis='x', rotation=15)
    st.pyplot(fig2)


# ------------------ ROW 2: TENURE KDE & CORR HEATMAP ------------------ #
st.markdown("---")
row2_col1, row2_col2 = st.columns(2)

# 3) Tenure Distribution by Churn Status (MATCH NOTEBOOK curves + Yes/No legend)
with row2_col1:
    st.subheader("Tenure Distribution by Churn Status")

    # temporary copy to map numbers to Yes/No (for legend)
    temp_df = df_model.copy()
    temp_df["ChurnLabel"] = temp_df["Churn"].map({0: "No", 1: "Yes"})

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    sns.kdeplot(
        data=temp_df,
        x="tenure",
        hue="ChurnLabel",     # now legend shows Yes/No
        fill=True,
        alpha=0.5,
        ax=ax3                # NOTE: default normalization (matches notebook)
    )

    ax3.set_title("Tenure Distribution by Churn Status")
    ax3.set_xlabel("tenure")
    ax3.set_ylabel("Density")
    st.pyplot(fig3)



# 4) Correlation Heatmap (MATCHES NOTEBOOK: numeric features of ENCODED MODEL DF)
with row2_col2:
    st.subheader("Correlation Heatmap (Encoded Numeric Features)")
    numeric_df_full = df_model.select_dtypes(include="number")
    corr_full = numeric_df_full.corr()

    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_full, cmap="coolwarm", center=0, ax=ax4)
    ax4.set_title("Correlation Heatmap")
    st.pyplot(fig4)


# ------------------ ROW 3: CONFUSION MATRIX & FEATURE IMPORTANCE ------------------ #
st.markdown("---")
row3_col1, row3_col2 = st.columns(2)

# 5) Confusion Matrix (from full model)
with row3_col1:
    st.subheader("Confusion Matrix (Random Forest)")
    fig5, ax5 = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", linewidths=0.5, ax=ax5)
    ax5.set_title("Confusion Matrix")
    ax5.set_xlabel("Predicted")
    ax5.set_ylabel("Actual")
    st.pyplot(fig5)

# 6) Top 15 Features Predicting Churn (from full model)
with row3_col2:
    st.subheader("Top 15 Features Predicting Churn")
    fig6, ax6 = plt.subplots(figsize=(7, 5))
    ax6.barh(top15.index[::-1], top15.values[::-1])
    ax6.set_title("Top 15 Features Predicting Churn")
    ax6.set_xlabel("Importance")
    ax6.set_ylabel("")
    st.pyplot(fig6)


# ------------------ RAW DATA ------------------ #
st.markdown("---")
with st.expander("Show Raw Filtered Data"):
    st.dataframe(filtered_df)



