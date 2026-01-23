# End-to-End-Mall-Customer-Segmentation-Predictor-using-Unsupervised-Learning
# 📊 Customer Segmentation Predictor (Streamlit App)

This project is an **interactive Customer Segmentation web application** built using **Streamlit** and **Unsupervised Machine Learning** techniques.  
It classifies customers into meaningful segments based on **Annual Income** and **Spending Score** using **K-Means** and **Hierarchical Clustering**.

---

## 🚀 Features

- Interactive Streamlit web interface  
- Customer segmentation using **K-Means Clustering**  
- Customer segmentation using **Hierarchical Clustering**  
- Real-time prediction for new customer data  
- Visual cluster representation with Matplotlib  
- Session-based validation dataset (current session only)  

---

## 🧠 Model Inputs

The prediction is based on the following customer attributes:

| Feature              | Description |
|----------------------|-------------|
| Annual Income (k$)   | Customer’s annual income |
| Spending Score       | Spending behavior score (1–100) |

---

## 🧩 Customer Segments

| Cluster | Segment Description |
|-------|---------------------|
| 0 | High Income & High Spending |
| 1 | High Income but Low Spending |
| 2 | Moderate Income & Moderate Spending |
| 3 | Low Income & High Spending |
| 4 | Low Income & Low Spending |

---

## 🛠 Tech Stack

- Python  
- Streamlit  
- NumPy  
- Pandas  
- Scikit-learn  
- Joblib  
- Matplotlib  

---

## 📊 Visualization

- Cluster distribution visualized using Matplotlib
- New customer highlighted with a star marker
### Separate visualizations for:
- K-Means clustering
- Hierarchical clustering

## 📌 Use Cases

- Customer profiling
- Targeted marketing strategies
- Personalized offers and promotions
- Business decision support
