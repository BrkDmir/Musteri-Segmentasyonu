# -*- coding: utf-8 -*-

import numpy 
import os

os.environ["OMP_NUM_THREADS"]="1"

from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

df = pd.read_csv("Avm_Musterileri.csv")
print(df.head())

plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Bazı sütun isimleri çok uzun olduğu için bunları kısaltma işlemi yapıyoruz

df.rename(columns={'Annual Income (k$)':'income'},inplace=True)
df.rename(columns={'Spending Score (1-100)':'score'},inplace=True)

# Normalization Process

scaler = MinMaxScaler()

scaler.fit(df[['income']])
df['income'] = scaler.transform(df[['income']])

scaler.fit(df[['score']])
df['score'] = scaler.transform(df[['score']])
print(df.head())

print(df.tail())

# Elbow yöntemiyle K değerinin bulunması işlemi.

kRange = range(1,11)

listDist = []

for k in kRange:
    kMeansModelim = KMeans(n_clusters=k)
    kMeansModelim.fit(df[['income','score']]) 
    listDist.append(kMeansModelim.inertia_)
    
    
    
plt.xlabel('K')
plt.ylabel('Distortion değeri (inertia)')
plt.plot(kRange,listDist)
plt.show()    

# K = 5 için bir K-Means modeli oluşturalım

kMeansModelim = KMeans(n_clusters=5)
yPredicted = kMeansModelim.fit_predict(df[['income','score']])
print(yPredicted)

df['cluster'] = yPredicted
print(df.head())

print(kMeansModelim.cluster_centers_)

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
df4 = df[df.cluster==3]
df5 = df[df.cluster==4]


plt.xlabel('income')
plt.ylabel('score')
plt.scatter(df1['income'],df1['score'],color='green')
plt.scatter(df2['income'],df2['score'],color='red')
plt.scatter(df3['income'],df3['score'],color='black')
plt.scatter(df4['income'],df4['score'],color='orange')
plt.scatter(df5['income'],df5['score'],color='purple')


plt.scatter(kMeansModelim.cluster_centers_[:,0], kMeansModelim.cluster_centers_[:,1], color='blue', marker='X', label='centroid')
plt.legend()
plt.show()

    


