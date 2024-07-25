# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 19:47:58 2023

@author: kaush
"""

#Using Normalize Function

#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

#Import Dataset
airline=pd.read_excel('EastWestAirlinesData.xlsx')
airline
airline.info()
airline2=airline.drop(['ID#'],axis=1)
airline2

# Normalize heterogenous numerical data 
airline2_norm=pd.DataFrame(normalize(airline2),columns=airline2.columns)
airline2_norm

# Create Dendrograms
plt.figure(figsize=(10, 7))  
dendograms=sch.dendrogram(sch.linkage(airline2_norm,'complete'))

# Create Clusters (y)
hclusters=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
hclusters
y=pd.DataFrame(hclusters.fit_predict(airline2_norm),columns=['clustersid'])
y['clustersid'].value_counts()

# Adding clusters to dataset
airline2['clustersid']=hclusters.labels_
airline2
airline2.groupby('clustersid').agg(['mean']).reset_index()

# Plot Clusters
plt.figure(figsize=(10, 7))  
plt.scatter(airline2['clustersid'],airline2['Balance'], c=hclusters.labels_) 