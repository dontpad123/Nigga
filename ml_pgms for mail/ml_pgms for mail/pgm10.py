import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings('ignore')

 data = pd.read_csv(r"Wisconsin Breast Cancer dataset.csv")

data.head()

data.shape

data.info()

data.nunique()

data.isnull().sum()

data.duplicated().sum()

df=data.drop(['id','Unnamed: 32'],axis=1)

df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})

df.describe().T


 #dropped the  Diagnosis (target) since clustering is unsupervised.
df.drop(columns=["diagnosis"], inplace=True)  

# Standardize the data
 scaler = StandardScaler()
 X_scaled = scaler.fit_transform(df)

#  Apply PCA for Dimensionality Reduction
 pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
 X_pca = pca.fit_transform(X_scaled)


 # Check explained variance ratio
 explained_variance = pca.explained_variance_ratio_
 total_explained_variance = np.sum(explained_variance)
 print(f"Variance explained by PC1: {explained_variance[0]:.4f}")
 print(f"Variance explained by PC2: {explained_variance[1]:.4f}")
 print(f"Total variance explained by first 2 components: {total_explained_variance:.4f}")


#Use the Elbow Method to determine the optimal number of clusters
wcss = []  # Within-Cluster Sum of Squares
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)  # Append the inertia (sum of squared distances)



 # Plot the Elbow Method Graph
plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker="o", linestyle="-")
plt.xlabel("Number of Clusters (k)") 
plt.ylabel("WCSS")
plt.title("Elbow Method to Find Optimal k") 
plt.show()


#Apply K-Means Clustering with the optimal k (usually where elbow occurs,  k=2)
optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200)s
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering after PCA")
plt.legend()
plt.show()
