#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 17:39:03 2020

@author: devil
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances  

distance = []
class Kmeans:
    def __init__(self, cluster_count, iteration_count = 200):
        self.cluster_count = cluster_count
        self.iteration_count = iteration_count    
    
    def assign_clusters(self, iter,model_data):
        dist_to_centroid =  pairwise_distances(model_data, self.centroids, metric = 'euclidean')
        if iter!=-1:
            distance.append(sum(np.min(dist_to_centroid,axis=1)))
        self.cluster_labels = np.argmin(dist_to_centroid, axis = 1)
        
        return  self.cluster_labels
    
    
    def fit_model(self, model_data):
        initial_centroids = np.random.permutation(model_data.shape[0])[:self.cluster_count]
        self.centroids = model_data[initial_centroids]
        for iter in range(self.iteration_count):
            self.cluster_labels = self.assign_clusters(iter,model_data)
            self.centroids = np.array([model_data[self.cluster_labels == i].mean(axis = 0) for i in range(self.cluster_count)])       
        return self  
    
    
class1 = pd.read_csv("non_linearly_seperable_data/Class1.txt", sep="\t", header=None)
class2 = pd.read_csv("non_linearly_seperable_data/Class2.txt", sep="\t", header=None)
class1 = class1.iloc[:,[0,1]]
class2 = class2.iloc[:,[0,1]]
#print(class1)
model = Kmeans(2,10)

# Spliting the model_data 
class1_train, class1_test = train_test_split(class1, test_size=0.3, random_state=42, shuffle=True)
class2_train, class2_test = train_test_split(class2, test_size=0.3, random_state=42, shuffle=True)

# Joining the train model_data as Class3
class3 = pd.concat([class1_train,class2_train], ignore_index=True)
# Joining the test model_data as Class3_test
class3_test= pd.concat([class1_test,class2_test], ignore_index=True)

Y_sklearn=class3.to_numpy()
fitted = model.fit_model(Y_sklearn)
prediction = model.assign_clusters(-1,class3_test.to_numpy())
#print(prediction)

count=0
for i in range(len(prediction)):
    if i<2000:
        if prediction[i]==1:
            count=count+1
    else:
        if prediction[i]==0:
            count=count+1
print("Accuracy is ", 100-count/4000)

classw=class3_test.to_numpy()
plt.scatter(classw[:, 0], classw[:, 1], c=prediction, s=5,cmap="Dark2")
plt.xlabel("Data Attribute1")
plt.ylabel("Data Attribute2")
plt.title("Given Data points.")
plt.show()


print("Convergence of GMM:")
plt.plot(distance)
plt.xlabel("No. of Iterations")
plt.ylabel("Loss")
plt.show()
