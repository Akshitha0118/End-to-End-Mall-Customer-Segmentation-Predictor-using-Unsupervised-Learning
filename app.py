import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="📊",
    layout="centered"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
.title { text-align: center; color: #2c3e50; }
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.result { font-size: 18px; font-weight: bold; color: #27ae60; }
</style>
""", unsafe_allow_html=True)

# =========================
# Load Models & Dataset
# =========================
kmeans_model = joblib.load("kmeans_model.pkl")
hierarchical_centroids = joblib.load("hierarchical_centroids.pkl")

dataset = pd.read_csv(r"C:\Users\Admin\Desktop\Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values
kmeans_labels = kmeans_model.predict(X)

# =========================
# Cluster Names & Colors
# =========================
cluster_names = {
    0: "High Income & High Spending ",
    1: "High Income but Low Spending " ,
    2: "Moderate Income & Moderate Spending ",
    3: "Low Income, High Spending ",
    4: "Low Income & Low Spending"
}

cluster_colors = {
    0: "red",
    1: "blue",
    2: "green",
    3: "orange",
    4: "purple"
}

# =========================
# Session State (NO HISTORY)
# =========================
if "validation_data" not in st.session_state:
    st.session_state.validation_data = []

def save_session_data(income, score, k_cluster, h_cluster):
    st.session_state.validation_data.append({
        "Annual Income (k$)": income,
        "Spending Score": score,
        "KMeans Cluster": f"{k_cluster} - {cluster_names[k_cluster]}",
        "Hierarchical Cluster": f"{h_cluster} - {cluster_names[h_cluster]}"
    })

# =========================
# App Title
# =========================
st.markdown("<h1 class='title'>📊 Customer Segmentation Predictor</h1>", unsafe_allow_html=True)

# =========================
# User Input
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)

annual_income = st.number_input("💰 Annual Income (k$)", min_value=5, step=5)
spending_score = st.number_input("🛍️ Spending Score (1–100)", min_value=1, step=1)

st.markdown("</div>", unsafe_allow_html=True)

input_point = np.array([[annual_income, spending_score]])

# =========================
# Prediction Buttons
# =========================
col1, col2 = st.columns(2)

# ---------- K-Means ----------
with col1:
    if st.button("🔵 Predict using K-Means"):
        k_cluster = kmeans_model.predict(input_point)[0]
        h_cluster = np.argmin(np.linalg.norm(hierarchical_centroids - input_point, axis=1))

        save_session_data(annual_income, spending_score, k_cluster, h_cluster)

        st.markdown(
            f"<div class='card result'>K-Means Cluster: {k_cluster} - {cluster_names[k_cluster]}</div>",
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots()
        for c in np.unique(kmeans_labels):
            ax.scatter(
                X[kmeans_labels == c, 0],
                X[kmeans_labels == c, 1],
                c=cluster_colors[c],
                label=f"Cluster {c}"
            )

        ax.scatter(annual_income, spending_score, c="black", s=300, marker="*", label="New Customer")
        ax.set_xlabel("Annual Income (k$)")
        ax.set_ylabel("Spending Score")
        ax.set_title("K-Means Cluster Visualization")
        ax.legend()
        st.pyplot(fig)

# ---------- Hierarchical ----------
with col2:
    if st.button("🟣 Predict using Hierarchical"):
        h_cluster = np.argmin(np.linalg.norm(hierarchical_centroids - input_point, axis=1))
        k_cluster = kmeans_model.predict(input_point)[0]

        save_session_data(annual_income, spending_score, k_cluster, h_cluster)

        st.markdown(
            f"<div class='card result'>Hierarchical Cluster: {h_cluster} - {cluster_names[h_cluster]}</div>",
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots()
        for c in np.unique(kmeans_labels):
            ax.scatter(
                X[kmeans_labels == c, 0],
                X[kmeans_labels == c, 1],
                c=cluster_colors[c],
                label=f"Cluster {c}"
            )

        ax.scatter(annual_income, spending_score, c="black", s=300, marker="*", label="New Customer")
        ax.set_xlabel("Annual Income (k$)")
        ax.set_ylabel("Spending Score")
        ax.set_title("Hierarchical Cluster Visualization")
        ax.legend()
        st.pyplot(fig)

# =========================
# Validation Dataset (SESSION ONLY)
# =========================
st.subheader("📂 User Validation Dataset (Current Session Only)")

if st.session_state.validation_data:
    st.dataframe(pd.DataFrame(st.session_state.validation_data))
else:
    st.info("No validation data recorded in this session.")

# =========================
# Footer
# =========================
st.markdown(
    "<p style='text-align:center;color:gray;'>Customer Segmentation | K-Means & Hierarchical</p>",
    unsafe_allow_html=True
)
