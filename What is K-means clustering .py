# -*- coding: utf-8 -*-
"""
Created on Sat May  8 02:08:26 2021

@author: abc
"""


import pandas as pd

#read out dataset
df = pd.read_excel('K_Means.xlsx')
print(df.head())

#Plot our dataset
import seaborn as sns
sns.regplot(x=df['X'], y=df['Y'], fit_reg=False) #fit_reg means do you want to divide it or not


######################################################################################################

#Apply K-means clustering

import pandas as pd

df = pd.read_excel('K_Means.xlsx')

#Apply k-means clustering using sklearn library
from sklearn.cluster import KMeans

#initiallized the k-means algorithm
kmeans = KMeans(n_clusters=3,init='k-means++', max_iter=300, n_init=10, random_state=0)

#fit the dataset
model = kmeans.fit(df)

#Predict our dataset
predicted_values = kmeans.predict(df)

#plot our dataset
from matplotlib import pyplot as plt

plt.scatter(df['X'], df['Y'], c=predicted_values, s=50, cmap='viridis')

#put centers on our clustring
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, c='black', alpha=0.5)



####################################################################################################


                                           #THANK YOU



