# import libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# Read the dataset 
dataset=pd.read_csv(r'C:\Users\Admin\Desktop\Mall_Customers.csv') 
X = dataset.iloc[:,[3,4]].values



# OMP Warning
import os
os.environ["OMP_NUM_THREADS"] = "1"



# Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()



#  K-Means Clustering 
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
y_kmeans = kmeans.fit_predict(X)


# KMeans Final data 
dataset['k_means_cluster']=y_kmeans



# Hierarchical Clustering – Dendrogram
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('dendogram')
plt.xlabel('customers')
plt.ylabel('eculidan distance')
plt.show()



# Agglomerative Clustering
hc=AgglomerativeClustering(n_clusters=5,metric='euclidean',linkage='ward')
y_hc=hc.fit_predict(X)


# Agglomerative Clustering Final data 
dataset['Agglomerative_cluster']=y_hc


import joblib
import numpy as np

# =========================
# Pickle K-Means Model
# =========================
joblib.dump(kmeans, "kmeans_model.pkl")

# =========================
# Pickle Hierarchical Data
# (store centroids because no predict())
# =========================
centroids = []
for i in range(5):
    centroids.append(X[y_hc == i].mean(axis=0))

centroids = np.array(centroids)

joblib.dump(centroids, "hierarchical_centroids.pkl")

print("✅ Pickle files created successfully")
