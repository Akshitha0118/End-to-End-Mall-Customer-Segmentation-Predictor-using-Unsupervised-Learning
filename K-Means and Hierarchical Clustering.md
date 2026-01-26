# 📊 Customer Segmentation Using K-Means and Hierarchical Clustering

## 📝 Project Overview
This project involves the segmentation of customers based on two important factors: their **annual income** and **spending score**. The objective is to group customers into clusters, which will allow businesses to target specific segments with tailored marketing strategies. In this project, both **K-Means** and **Hierarchical Clustering** techniques are applied to identify customer segments.

---

## 📂 Dataset Overview
The dataset consists of **200 records** with the following columns:

1. **CustomerID**: A unique identifier for each customer.
2. **Gender**: The gender of the customer (Male/Female).
3. **Age**: The age of the customer.
4. **Annual Income (k$)**: The annual income of the customer in thousand dollars.
5. **Spending Score (1-100)**: A score assigned by the mall, based on customer spending behavior.

### 🔍 Data Sample:
| CustomerID | Gender | Age | Annual Income (k$) | Spending Score (1-100) |
|------------|--------|-----|--------------------|------------------------|
| 1          | Male   | 19  | 15                 | 39                     |
| 2          | Male   | 21  | 15                 | 81                     |
| 3          | Female | 20  | 16                 | 6                      |
| 4          | Female | 23  | 16                 | 77                     |
| 5          | Female | 31  | 17                 | 40                     |

---

## 🔧 Data Preprocessing
Before performing clustering, the following preprocessing steps were carried out:

1. **Selecting Relevant Features**:  
   We focused on two columns: `Annual Income (k$)` and `Spending Score (1-100)` for clustering. These features are used to group customers based on their income and spending behavior.

2. **Handling Missing Values**:  
   The dataset was checked for any missing or infinite values, and none were found.

3. **Standardization**:  
   In this case, standardization was not necessary, as the features are already on a comparable scale.

---

## 📉 Elbow Method for Optimal Cluster Selection (K-Means)
The **Elbow Method** was used to determine the optimal number of clusters for the K-Means algorithm by plotting the **WCSS (Within-Cluster Sum of Squares)** for different values of `k` (number of clusters).

### Elbow Method Plot:
The plot of **WCSS** vs. the number of clusters indicated that **5 clusters** is the optimal choice for segmentation.

![image](https://github.com/user-attachments/assets/15ef014b-cdda-4c2b-9bba-6f5e7c80fe37)

---

## 🔵 K-Means Clustering
The **K-Means** algorithm was applied to the dataset with **5 clusters**. This technique grouped customers into 5 distinct segments based on their annual income and spending score.

### 📊 K-Means Clustering Visualization:
A scatter plot was created to visualize the clusters, with each customer assigned to one of the 5 clusters. The cluster centroids were also marked on the plot.

![K-Means Clustering](https://github.com/user-attachments/assets/5a3e9d7b-348f-4cbe-9230-2f99cb028c17)

---

## 🏔️ Hierarchical Clustering (Agglomerative)
Along with K-Means, **Hierarchical Clustering** was also performed using **Agglomerative Clustering**. A **dendrogram** was plotted to visualize the hierarchy of clusters, and the optimal number of clusters (5) was determined.

![Agglomerative Clustering](https://github.com/user-attachments/assets/c52df549-1e1f-4f79-ae0a-4ff5ba878937)


### 📊 Hierarchical Clustering Visualization:
A scatter plot was generated similar to the K-Means visualization, showing the 5 clusters formed by **Hierarchical Clustering**.

![image](https://github.com/user-attachments/assets/f42c9526-74e5-4e6a-afe1-9ccdce97684d)

---

## 🔑 Key Insights
- **Customer Segments**: The 5 clusters represent distinct segments of customers based on their income and spending score, allowing businesses to target specific customer groups more effectively.
- **Comparison of Methods**: Both **K-Means** and **Hierarchical Clustering** produced consistent groupings, indicating the reliability of these clustering methods.

---

## 🎯 Conclusion
This project successfully applied **clustering techniques** to segment customers into distinct groups based on their spending behavior and income. These insights can help businesses improve their marketing strategies and cater to specific customer segments more efficiently.

---
