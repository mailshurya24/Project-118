import pandas as pd
import statistics as st
import numpy as np
import plotly.express as pe
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sb

data= pd.read_csv("stars.csv")
size=data["Size"].tolist()
light=data["Light"].tolist()

#Plotting the data in a scatter plot to see if we can visually tell the clusters apart from one another.commands.Cog.listener()
scatter = pe.scatter(x=size,y=light, title="Original Scatter Plot")
scatter.show()

#Finding the best K value using the WCSS perimeter.
X=data.iloc[:,[0,1]].values
WCSS=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init="k-means++", random_state=1)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)
print("Sum of squares within clusters:", WCSS)

#Finding the K value using the Elbow Method
plt.figure(figsize=(7,7))
sb.lineplot(range(1,11), WCSS, marker="o", color="purple")
plt.title("Elbow Plot")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
print("Launching Elbow plot...")
plt.show()
print("Launched! Now, for the next part of the program to run, please close the elbow plot.")

#Putting the data into three clusters
kmeans=KMeans(n_clusters=3, init="k-means++", random_state=1)
y_kmeans= kmeans.fit_predict(X)
plt.figure(figsize=(8,8))
#Plotting cluster 1
sb.scatterplot(X[y_kmeans==0,0], X[y_kmeans==0,1], color="red", label="Cluster 1")

#Plotting cluster 2
sb.scatterplot(X[y_kmeans==1,0], X[y_kmeans==1,1], color="blue", label="Cluster 2")

#Plotting cluster 3
sb.scatterplot(X[y_kmeans==2,0], X[y_kmeans==2,1], color="green", label="Cluster 2")

#Plotting the centroid
sb.scatterplot(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color="turquoise", s=150, marker=",", label="Centroids")

plt.title("Clusters For Stars VS Light Data")
plt.legend()
plt.grid(False)
plt.xlabel("Stars")
plt.ylabel("Light")
print("Getting final graph....")
plt.show()
print("Done!")

#End of program
