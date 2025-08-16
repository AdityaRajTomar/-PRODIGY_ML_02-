import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load customer purchase history data
# Replace this with your actual dataset
# For demonstration, let's simulate some data:
data = {
    'CustomerID': range(1, 11),
    'Annual_Spend': [1500, 2400, 1800, 3000, 5000, 1200, 800, 2200, 3500, 2700],
    'Visit_Frequency': [10, 15, 12, 20, 25, 8, 5, 13, 22, 17]
}
df = pd.DataFrame(data)

# Step 2: Feature selection
features = df[['Annual_Spend', 'Visit_Frequency']]

# Step 3: Data normalization
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 4: Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Step 5: Visualize the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Annual_Spend', y='Visit_Frequency', hue='Cluster', palette='viridis', s=100)
plt.title('Customer Segmentation Based on Purchase History')
plt.xlabel('Annual Spend')
plt.ylabel('Visit Frequency')
plt.grid(True)
plt.show()
