# Importing required libraries
# Using train_test_split for spliting the data randomly
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Opening and reading the data files 
class1 = pd.read_csv("real_world_data/class1.txt",sep=" ",header=None)
class2 = pd.read_csv("real_world_data/class2.txt",sep=" ",header=None)
class3 = pd.read_csv("real_world_data/class3.txt",sep=" ",header=None)

# Third column NAN eliminated
class1 = class1[[0,1]]
class2 = class2[[0,1]]
class3 = class3[[0,1]]

# Splitting into test and train, test = 30% and train 70%
class1_train, class1_test = train_test_split(class1, test_size=0.3, random_state=42, shuffle=True)
class2_train, class2_test = train_test_split(class2, test_size=0.3, random_state=42, shuffle=True)
class3_train, class3_test = train_test_split(class3, test_size=0.3, random_state=42, shuffle=True)

# Scatter plot of given data
print("Scatter plot of given data :")
plt.scatter(class1[[0]],class1[[1]],label="Class1",c="blue")
plt.scatter(class2[[0]],class2[[1]],label="Class2",c="orange")
plt.scatter(class3[[0]],class3[[1]],label="Class3",c="green")
plt.xlabel("Attribute1")
plt.ylabel("Attribute2")
plt.title("Data")
plt.show()
print(80*"*")

# Scatter plot of Training data
print("Scatter plot of Training data :")
plt.scatter(class1_train[[0]],class1_train[[1]],label="Class1")
plt.scatter(class2_train[[0]],class2_train[[1]],label="Class2")
plt.scatter(class3_train[[0]],class3_train[[1]],label="Class3")
plt.xlabel("Attribute1")
plt.ylabel("Attribute2")
plt.title("Training data")
plt.show()
print(80*"*")

# Scatter plot of Test data
print("Scatter plot of Test data :")
plt.scatter(class1_test[[0]],class1_test[[1]],label="Class1")
plt.scatter(class2_test[[0]],class2_test[[1]],label="Class2")
plt.scatter(class3_test[[0]],class3_test[[1]],label="Class3")
plt.xlabel("Attribute1")
plt.ylabel("Attribute2")
plt.title("Test data")
plt.show()
print(80*"*")

# Function for calculating mean of data
def mean(data):
    
  cols = len(data.columns)
  mu = np.zeros(cols)
  for i in range(len(data)):
    for j in range(cols):
      mu[j] = mu[j] + data.iloc[i][j]
  mu = mu/len(data)
  return mu

# Function for calculating covariance
def covar(data, mean):
  n = len(data)-1
  cols = len(data.columns)
  covar = np.zeros((cols,cols))
  for i in range(n):
    covar = covar + np.outer(data.iloc[i]-mean, np.transpose(data.iloc[i]-mean))
  covar = covar/n
  return covar

# Discriminant function for classifying
def classifier(n, mu, covar, vec, total_samp):
  x = (-1/2)*(np.matmul(np.transpose(vec - mu), np.linalg.inv(covar)))
  temp1 = np.matmul(x, vec - mu)
  temp2 = (-1/2)*np.log(np.linalg.det(covar))
  temp3 = np.log(n/total_samp)
  classifier = temp1 + temp2 + temp3
  return classifier


# Finding mean of training data
mean1 = mean(class1_train)
mean2 = mean(class2_train)
mean3 = mean(class3_train)

# Finding covariance of training data
cov1 = covar(class1_train, mean1)
cov2 = covar(class2_train, mean2)
cov3 = covar(class3_train, mean3)

# Corrected Predicted till now
corr_pred = 0

# Total training samples
total_n = len(class1_train)+len(class2_train)+len(class3_train)

# Classifying using Bayes classifier
for i in range(len(class1_test)):
    classifier1 = classifier(len(class1_train),mean1,cov1,class1_test.iloc[i],total_n)
    classifier2 = classifier(len(class2_train),mean2,cov2,class1_test.iloc[i],total_n)
    classifier3 = classifier(len(class3_train),mean3,cov3,class1_test.iloc[i],total_n)
    if(classifier1 >= classifier2 and classifier1 >= classifier3):
        corr_pred = corr_pred+1
        
for i in range(len(class2_test)):
    classifier1 = classifier(len(class1_train),mean1,cov1,class2_test.iloc[i],total_n)
    classifier2 = classifier(len(class2_train),mean2,cov2,class2_test.iloc[i],total_n)
    classifier3 = classifier(len(class3_train),mean3,cov3,class2_test.iloc[i],total_n)
    if(classifier2 >= classifier1 and classifier2 >= classifier3):
        corr_pred = corr_pred+1

for i in range(len(class3_test)):
    classifier1 = classifier(len(class1_train),mean1,cov1,class3_test.iloc[i],total_n)
    classifier2 = classifier(len(class2_train),mean2,cov2,class3_test.iloc[i],total_n)
    classifier3 = classifier(len(class3_train),mean3,cov3,class3_test.iloc[i],total_n)
    if(classifier3 >= classifier1 and classifier3 >= classifier1):
        corr_pred = corr_pred+1

# Accuracy of the Bayes classifier
print()
print("*** Bayes Classifier - Real World data***")
print()
print("Total correctly predicted points : ", corr_pred)
print("Total test data points : ", len(class1_test)+len(class2_test)+len(class3_test))
print("Accuracy of classifier : ",corr_pred/(len(class1_test)+len(class2_test)+len(class3_test)))
print()
print(80*"*")