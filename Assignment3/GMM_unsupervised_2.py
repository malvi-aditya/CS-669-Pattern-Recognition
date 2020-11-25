# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal as mvn

# Opening and reading the data files  
class1 = pd.read_csv("data_2/Class1.txt", sep="\t", header=None)
class2 = pd.read_csv("data_2/Class2.txt", sep="\t", header=None)

# Third column NAN eliminated
class1 = class1[[0,1]]
class2 = class2[[0,1]]

plt.scatter(class1[[0]], class1[[1]], label = "Class1", c = "green")
plt.scatter(class2[[0]], class2[[1]], label = "Class2", c = "orange")
plt.xlabel("Data Attribute1")
plt.ylabel("Data Attribute2")
plt.title("Given Data points.")


a = []
b = []

# Define class of a GMM
class GMM:
    
    # init function
    def __init__(self, cluster_count, iteration_count):
        self.cluster_count = cluster_count 
        self.iteration_count = iteration_count
    
    # Implement k-means to find starting parameter values
    def parameter_initialisation_kmeans(self, arr):
        
        n_clusters = self.cluster_count
        kmeans = KMeans(n_clusters= n_clusters, init='k-means++', max_iter=100)
        prediction = kmeans.fit_predict(arr)
        dim = arr.shape[1] 
        labels = np.unique(prediction)
        self.initial_mean = np.zeros((self.cluster_count, dim))
        self.initial_cov = np.zeros((self.cluster_count, dim, dim))
        self.initial_pi = np.zeros(self.cluster_count)
        
        counter=0
        for label in labels:
            ids = np.where(prediction == label) 
            self.initial_pi[counter] = len(ids[0]) / arr.shape[0]
            self.initial_mean[counter,:] = np.mean(arr.iloc[ids])
            x_minus_mu = arr.iloc[ids] - self.initial_mean[counter,:]
            Nk = arr.iloc[ids].shape[0]
            self.initial_cov[counter,:, :] = np.dot(self.initial_pi[counter] * x_minus_mu.T, x_minus_mu) / Nk
            counter+=1
        return (self.initial_mean, self.initial_cov, self.initial_pi)
    
    # Define function for E step of EM algorithm
    def E_Step(self, arr, pi, mean, sigma):
        self.gamma_function = np.zeros((arr.shape[0],self.cluster_count))
        self.mean = self.initial_mean
        self.pi = self.initial_pi
        self.sigma = self.initial_cov
        
        # Posterior Distribution for gamma_function using Bayes Rule
        for c in range(self.cluster_count):
            self.gamma_function[:,c] = self.pi[c] * mvn.pdf(arr, self.mean[c,:], self.sigma[c])
        gamma_function_norm = np.sum(self.gamma_function, axis=1)[:,np.newaxis]
        self.gamma_function /= gamma_function_norm
        return self.gamma_function
    
    # Define function for M step of EM algorithm
    def M_Step(self, arr, gamma_function):
        # responsibilities    
        for c in range(self.cluster_count):
            x = arr - self.mean[c, :]
            gamma_function_diag = np.diag(self.gamma_function[:,c])
            gamma_function_diag = np.matrix(gamma_function_diag)
            sigma_c = x.T @ gamma_function_diag @ x
            self.sigma[c,:,:]=(sigma_c) / np.sum(self.gamma_function, axis = 0)[c]    
        return (self.pi, self.mean, self.sigma)

     # reference taken from bishop and towardsdatascience..
    # Define function for computing log likelihood 
    def Loss_function(self, arr, pi, mean, sigma):
        cluster_count = self.gamma_function.shape[1]
        self.loss = np.zeros((arr.shape[0], cluster_count))

        for c in range(cluster_count):
            dist = mvn(self.mean[c], self.sigma[c],allow_singular=True)
            self.loss[:,c]=self.gamma_function[:,c]*(np.log(self.pi[c]+0.00001)+dist.logpdf(arr)-np.log(self.gamma_function[:,c]+0.000001))
        self.loss = np.sum(self.loss)
        return self.loss
    
    
    def assign_cluster(self, arr):
        labels = np.zeros((arr.shape[0], self.cluster_count))
        for c in range(self.cluster_count):
            labels [:,c] = self.pi[c] * mvn.pdf(arr, self.mean[c,:], self.sigma[c])
        labels  = labels.argmax(1)
        return labels

    
    # Function to compute the E-step and M-step and calculate the loss
    def fit_model(self, arr):
        self.mean, self.sigma, self.pi = self.parameter_initialisation_kmeans(arr)

        for run in range(self.iteration_count):  
            self.gamma_function  = self.E_Step(arr, self.mean, self.pi, self.sigma)
            self.pi, self.mean, self.sigma = self.M_Step(arr, self.gamma_function)
            loss = self.Loss_function(arr, self.pi, self.mean, self.sigma)        
            a.append(run)
            b.append(loss)
        return self
    
 # referenced from stackoverflow.   
# Define function to draw ellipse with given position and covariance
def draw_ellipse(position, covariance, ax=None, **kwargs):
    
    ax = ax or plt.gca()
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))
        

# Initialize the model by above GMM class
model = GMM(2, iteration_count = 30)
class3 = pd.concat([class1,class2], ignore_index=True)

# Fit the model on data 
fitted_values = model.fit_model(class3)
predicted_values = model.assign_cluster(class3)
        

# plot all the figures 
plt.figure(figsize = (11,9))
plt.ylim(ymin=-20)
plt.ylim(ymax=30)
plt.xlim(xmin=-15)
plt.xlim(xmax=45)
w_factor = 0.2 / model.pi.max()
for pos, covar, w in zip(model.mean, model.sigma, model.pi):
    draw_ellipse(pos, covar, alpha=w * w_factor)

print("Scatter plot of given data :")
classw=class3.to_numpy()
plt.scatter(classw[:, 0], classw[:, 1], c=predicted_values, s=5,cmap="Dark2")
plt.xlabel("Data Attribute1")
plt.ylabel("Data Attribute2")
plt.title("Given Data points.")
plt.show()

# Plot convergence
print("Convergence of GMM:")
plt.plot(a,b)
plt.xlabel("No. of Iterations")
plt.ylabel("Loss value")
plt.show()