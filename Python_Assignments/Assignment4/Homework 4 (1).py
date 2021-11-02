# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:33:16 2021

@author: 16319
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("student.csv")

#Use hierarchical clustering with average linkage and Euclidean distanceto cluster this data set.

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

Euc_dist = pdist(data, 'euclidean')
hc_linkage = linkage(Euc_dist, "average")
print(hc_linkage)


#2.  Cut the dendrogram at a height that results in three distinct clusters.Which observations belong to which clusters?
from scipy.cluster.hierarchy import dendrogram

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    hc_linkage,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

from scipy.cluster.hierarchy import cut_tree
print(cut_tree(hc_linkage, n_clusters = 3).T)

plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    hc_linkage,
    leaf_rotation=90.,  
    leaf_font_size=8.,
)
plt.show()
print(cut_tree(hc_linkage, n_clusters = 3).T)

#3.  Use  the  elbow  method  to  find  the  number,Kof  clusters  needed  toperformK−means clustering.

from sklearn.cluster import KMeans

cluster_range = np.arange(2,11, 1)
inertia = []

for i in cluster_range:
    kmeans = KMeans(init = 'k-means++', n_clusters = i, n_init = 20, random_state =0).fit(data)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(cluster_range, inertia, marker = 'o')
plt.show()

print(inertia)
#Elbow at n_cluster = 3
#Now performing K-means clustering with K = 3
kmeans = KMeans(n_clusters =3 , random_state = 123).fit(data)
print(kmeans.labels_)

data = pd.DataFrame(data)
plt.figure(figsize=(6,5))
plt.scatter(data.iloc[:,0], data.iloc[:,1], c = kmeans.labels_, cmap = plt.cm.bwr) 
plt.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1],
            marker = '*', 
            s = 150,
            color = 'cyan', 
            label = 'Centers')
plt.legend(loc = 'best')
plt.xlabel('X0')
plt.ylabel('X1')

compare1 = cut_tree(hc_linkage, n_clusters = 3)
compare2 = kmeans.labels_

result = np.zeros(len(compare1))
for i in range (len(compare1)):
    if (compare1[i] == compare2[i]):
        result[i] = 1
    else:
        result[i] = 0
    
comparison = np.mean(result)
print("Percent of equivalent clusters:", comparison*100)


#5.  Perform PCA on the observations and plot the first two principal com-ponent score vectors.

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

X = pd.DataFrame(scale(data), index=data.index, columns = data.columns)
pca_loadings = pd.DataFrame(PCA().fit(X).components_.T, index=data.columns, columns =['V1','V2','V3','V4'])
pca_loadings

#dot_prod = about 1 to check loading vectors
dot_prod = pca_loadings["V1"].dot(pca_loadings["V1"])

pca = PCA()
data_plot = pd.DataFrame(pca.fit_transform(X), columns = ['PC1', 'PC2', 'PC3', 'PC4'], index=X.index)
fig, ax1 = plt.subplots(figsize=(9,7))

ax1.set_xlim(-3.5,3.5)
ax1.set_ylim(-3.5,3.5)

#Plotting PC's #1 #2
for i in data_plot.index:
     ax1.annotate(i, (data_plot.PC1.loc[i], -data_plot.PC2.loc[i]), ha='center')

#Reference lines
ax1.hlines(0,-3.5,3.5, linestyles='dotted', colors = 'grey')
ax1.vlines(0,-3.5,3.5, linestyles='dotted', colors='grey')

ax1.set_xlabel('First Principal Component')
ax1.set_ylabel('Second Principal Component')

#Now performing K-means clustering with K = 3 on score vectros
PC1_Scores = np.zeros(shape = [len(data_plot), 1])
PC2_Scores = np.zeros(shape = [len(data_plot), 1])
for i in range(len(data_plot)):
    PC1_Scores[i] = data_plot.PC1.loc[i]
    PC2_Scores[i] = data_plot.PC2.loc[i]
    
PC_Scores_df = pd.DataFrame(data = PC1_Scores, columns = ["PC1"])
PC_Scores_df["PC2"] = PC2_Scores
    

kmeans1 = KMeans(n_clusters =3 , random_state = 123).fit(PC_Scores_df)
print(kmeans1.labels_)

plt.figure(figsize=(6,5))
plt.scatter(data.iloc[:,0], data.iloc[:,1], c = kmeans1.labels_, cmap = plt.cm.bwr) 
plt.scatter(kmeans1.cluster_centers_[:, 0], 
            kmeans1.cluster_centers_[:, 1],
            marker = '*', 
            s = 150,
            color = 'cyan', 
            label = 'Centers')
plt.legend(loc = 'best')
plt.xlabel('X0')
plt.ylabel('X1')

compare_1 = kmeans.labels_
compare_2 = kmeans1.labels_

result_ = np.zeros(len(compare_1))
for i in range (len(compare_1)):
    if (compare_1[i] == compare_2[i]):
        result_[i] = 1
    else:
        result_[i] = 0
    
comparison_ = np.mean(result)
print("Percent of equivalent clusters:", comparison_*100)

#1.  Determine the proportion of (standardized) sample variance due to thefirst two sample principal components, the first three sample principalcomponents.
from sklearn.preprocessing import StandardScaler
#Data scaled earlier in the problem
pca2 = PCA(n_components = 2)
data_plot_2 = pd.DataFrame(pca2.fit_transform(X), columns = ['PC1', 'PC2'])
print("Proportion of Variance of first 2 components:", pca2.explained_variance_ratio_)

pca3 = PCA(n_components = 3)
data_plot_3 = pd.DataFrame(pca3.fit_transform(X), columns = ['PC1', 'PC2', 'PC3'])
print("Proportion of Varince of first 3 components:", pca3.explained_variance_ratio_)

plt.figure(figsize=(7,5))

plt.plot([1,2,3,4], pca.explained_variance_ratio_, '-o', label='Individual component')
plt.plot([1,2,3,4], np.cumsum(pca.explained_variance_ratio_), '-s', label='Cumulative')

plt.ylabel('Proportion of Variance Explained')
plt.xlabel('Principal Component')
plt.xlim(0.75,4.25)
plt.ylim(0,1.05)
plt.xticks([1,2,3,4])
plt.legend(loc=2);

#Choose first 2 principal components
