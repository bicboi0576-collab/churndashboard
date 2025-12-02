import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")
st.title("📊 Telco Customer Churn Dashboard")


# ------------------ LOAD & PREP DATA ------------------ #
@st.cache_data
def load_and_prepare():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Drop ID, clean TotalCharges
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    # Encode all object columns (including Churn, Contract, etc.)
    le_dict = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Train / test split for model
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest model
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Feature importance
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top15 = importances.sort_values(ascending=False).head(15)

    return df, cm, top15

df, cm, top15 = load_and_prepare()


# ------------------ CHART 1: CHURN COUNT ------------------ #
st.markdown("### 1. Churn vs Non-Churn Count")
fig1, ax1 = plt.subplots(figsize=(5, 4))
sns.countplot(data=df, x="Churn", ax=ax1)
ax1.set_title("Churn vs Non-Churn Count")
ax1.set_xlabel("Churn")
ax1.set_ylabel("Count")
st.pyplot(fig1)


# ------------------ CHART 2: TENURE KDE BY CHURN ------------------ #
st.markdown("### 2. Tenure Distribution by Churn Status")
fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.kdeplot(
    data=df,
    x="tenure",
    hue="Churn",
    fill=True,
    alpha=0.5,
    common_norm=False,
    ax=ax2
)
ax2.set_title("Tenure Distribution by Churn Status")
ax2.set_xlabel("tenure")
ax2.set_ylabel("Density")
st.pyplot(fig2)


# ------------------ CHART 3: CHURN BY CONTRACT TYPE ------------------ #
st.markdown("### 3. Churn Rate by Contract Type")
fig3, ax3 = plt.subplots(figsize=(7, 4))
sns.countplot(data=df, x="Contract", hue="Churn", ax=ax3)
ax3.set_title("Churn Rate by Contract Type")
ax3.set_xlabel("Contract")
ax3.set_ylabel("count")
st.pyplot(fig3)


# ------------------ CHART 4: CORRELATION HEATMAP ------------------ #
st.markdown("### 4. Correlation Heatmap")
numeric_df = df.select_dtypes(include="number")
corr = numeric_df.corr()

fig4, ax4 = plt.subplots(figsize=(12, 8))
sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax4)
ax4.set_title("Correlation Heatmap")
st.pyplot(fig4)


# ------------------ CHART 5: CONFUSION MATRIX ------------------ #
st.markdown("### 5. Confusion Matrix (Random Forest)")
fig5, ax5 = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", linewidths=0.5, ax=ax5)
ax5.set_title("Confusion Matrix")
ax5.set_xlabel("Predicted")
ax5.set_ylabel("Actual")
st.pyplot(fig5)


# ------------------ CHART 6: FEATURE IMPORTANCE ------------------ #
st.markdown("### 6. Top 15 Features Predicting Churn")
fig6, ax6 = plt.subplots(figsize=(8, 6))
ax6.barh(top15.index[::-1], top15.values[::-1])
ax6.set_title("Top 15 Features Predicting Churn")
ax6.set_xlabel("Importance")
ax6.set_ylabel("")
st.pyplot(fig6)


# ------------------ RAW DATA ------------------ #
st.markdown("---")
with st.expander("Show Raw Data"):
    st.dataframe(df)
