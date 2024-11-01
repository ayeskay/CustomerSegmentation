# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the data
data = pd.read_csv('pds_alt_dataset.csv')
data = data.dropna()  # Remove rows with missing data

# Data preprocessing and feature engineering
# (Add your preprocessing steps here, similar to your notebook)
data['Age'] = 2024 - data['Year_Birth']
data["TotalSpend"] = data["MntWines"] + data["MntFruits"] + data["MntMeatProducts"] + data["MntFishProducts"] + data["MntSweetProducts"] + data["MntGoldProds"]
data["Children"] = data["Kidhome"] + data["Teenhome"]
data["Is_Parent"] = np.where(data.Children > 0, 1, 0)
data["Education"] = data["Education"].replace({"Basic": "Undergraduate", "2n Cycle": "Undergraduate", "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate"})
to_drop = data[["Dt_Customer", "Year_Birth", "ID"]]
data = data.drop(to_drop, axis=1)

# Streamlit layout
st.title('Customer Segmentation Analysis')

# Display data info
if st.checkbox('Show raw data'):
    st.write(data)

# Pair plot
st.subheader("Pair Plot Of Selected Features")
if st.checkbox('Show pair plot'):
    To_Plot = ["Income", "Recency", "Age", "TotalSpend", "Is_Parent"]
    pair_plot = sns.pairplot(data[To_Plot], hue="Is_Parent", palette=["#344E41", "#BC4749"])
    st.pyplot(pair_plot)

# Income distribution
st.subheader("Income Distribution")
fig, ax = plt.subplots()
sns.histplot(data['Income'], kde=True, color="#344E41", ax=ax)
st.pyplot(fig)

# Average Total Spend by Marital Status
st.subheader("Average Total Spend by Marital Status")
fig, ax = plt.subplots()
sns.barplot(x='Marital_Status', y='TotalSpend', data=data, palette="Blues_d", ax=ax)
st.pyplot(fig)

# KMeans Clustering
st.subheader("K-Means Clustering Analysis")
optimal_k = st.number_input('Select the number of clusters (k)', min_value=1, max_value=10, value=4)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
data['cluster'] = kmeans.fit_predict(data.select_dtypes(include=[np.number]))

# Plot PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(data.select_dtypes(include=[np.number]))
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=data['cluster'], cmap='inferno', edgecolor='k', s=50)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
plt.colorbar(scatter, ax=ax, label='Cluster Label')
st.pyplot(fig)

# Silhouette Score
silhouette_avg = silhouette_score(data.select_dtypes(include=[np.number]), data['cluster'])
st.write(f'Silhouette Score: {silhouette_avg:.2f}')

# Summary of clusters
st.subheader("Cluster Summary")
cluster_summary = data.groupby('cluster').agg(
    count=('cluster', 'size'),
    mean_income=('Income', 'mean'),
    median_income=('Income', 'median'),
    mean_age=('Age', 'mean'),
    median_age=('Age', 'median'),
)
st.write(cluster_summary)

# Save analysis to CSV
if st.button('Save Cluster Summary'):
    cluster_summary.to_csv('cluster_summary.csv', index=True)
    st.success('Cluster summary saved as "cluster_summary.csv".')
