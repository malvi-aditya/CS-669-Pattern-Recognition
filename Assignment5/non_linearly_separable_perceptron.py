# Importing required libraries
# Using train_test_split for spliting the data randomly
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Define the class of a perceptron
class Perceptron:
    
    # Define the init function
    def __init__(self, weights, bias=1, learning_rate=0.3):
       
        #Initialize all variables
        self.weights = np.array(weights)
        self.bias = bias
        self.learning_rate = learning_rate
        
    # Define the step function/ sign function    
    @staticmethod
    def sign_function(x):
        
        # If x < 0 then class1 else class2
        if  x <= 0:
            return 0
        else:
            return 1
        
    # Define the call function    
    def __call__(self, arr):
        
        # Classify a point
        bias_arr = [self.bias]
        arr = np.concatenate( (arr, bias_arr) )
        result = self.weights @ arr
        return Perceptron.sign_function(result)
    
    # Define function to update weights 
    def update_weights(self, target_result, arr):
        
        # If input not numpy array then convert it to np array
        if type(arr) != np.ndarray:
            arr = np.array(arr)  
            
        # classify the point
        calculated_result = self(arr)
        
        # If point misclassified then update the weights
        error = target_result - calculated_result
        if error != 0:
            bias_arr = [self.bias]
            arr = np.concatenate( (arr, bias_arr) )
            correction = error * arr * self.learning_rate
            self.weights += correction
            
    # Define function to test the model        
    def test_model(self, data, labels):
        
        # Variable to store count of correct classifications
        count = 0
        
        # If point not misclassified count+=1
        for sample, label in zip(data, labels):
            result = self(sample) # predict
            if result == label:
                count += 1
        return count
    
    
# Opening and reading the data files  
class1 = pd.read_csv("non_linearly_separable_data/Class1.txt", sep="\t", header=None)
class2 = pd.read_csv("non_linearly_separable_data/Class2.txt", sep="\t", header=None)

# Third column NAN eliminated
class1 = class1[[0,1]]
class2 = class2[[0,1]]

# Splitting into test and train, test = 30% and train 70%
class1_train, class1_test = train_test_split(class1, test_size=0.3, random_state=42, shuffle=True)
class2_train, class2_test = train_test_split(class2, test_size=0.3, random_state=42, shuffle=True)

# Scatter plot of given data
print("Scatter plot of given data :")
plt.scatter(class1[[0]], class1[[1]], label = "Class1", c = "blue")
plt.scatter(class2[[0]], class2[[1]], label = "Class2", c = "orange")
plt.xlabel("Attribute1")
plt.ylabel("Attribute2")
plt.title("Given Data")
plt.show()
print(80*"*")

# Scatter plot of Training data
print("Scatter plot of Training data : ")
plt.scatter(class1_train[[0]], class1_train[[1]], label="Class1")
plt.scatter(class2_train[[0]], class2_train[[1]], label="Class2")
plt.xlabel("Attribute1")
plt.ylabel("Attribute2")
plt.title("Training data")
plt.show()
print(80*"*")

# Scatter plot of Test data
print("Scatter plot of Test data : ")
plt.scatter(class1_test[[0]], class1_test[[1]], label="Class1")
plt.scatter(class2_test[[0]], class2_test[[1]], label="Class2")
plt.xlabel("Attribute1")
plt.ylabel("Attribute2")
plt.title("Test data")
plt.show()
print(80*"*")


# Initialize the model
model = Perceptron(weights=[1.0, 1.0, 1.0], learning_rate=0.8)

# Prepare the training data 
learn_data = pd.concat([class1_train, class2_train], axis = 0)
learn_data = learn_data.to_numpy()

# Prepare the training labels
l1 = np.zeros((700,), dtype=int)
l2 = np.ones((700,), dtype=int)
learn_labels = np.concatenate((l1, l2), axis = 0)

# Run over all samples n_iter times and update the weights
n_iter = 500
for i in range(n_iter):    
    for sample, label in zip(learn_data, learn_labels):
        model.update_weights(label, sample)
    
# Prepare the testing data
test_data = pd.concat([class1_test, class2_test], axis = 0)
test_data = test_data.to_numpy()

# Prepare the testing labels  
t1 = np.zeros((300,), dtype=int)
t2 = np.ones((300,), dtype=int)
test_labels = np.concatenate((t1, t2), axis = 0)


# Test the model
print()
print("Testing the model: ")
print()
evaluation = model.test_model(test_data, test_labels)
print("Correctly classified: ", evaluation)
print("Incorrectly classified: ", 600 - evaluation)
print("Acccuracy: ", round(evaluation/600,4))
print()
print(80*"*")


# Print details of the decision boundary
print()
print("Details of decision boundary: ")
print()
X = np.arange(np.min(learn_data[:,0]), np.max(learn_data[:,0]))
m = -model.weights[0] / model.weights[1]
c = -model.weights[2] / model.weights[1]
print("Slope of decision boundary:", m)
print("Y-Intercept of decision boundary:", c)
print()
print(80*"*")


# Plot the decision boundary and data
print("Plot of decision boundary: ")
fig, ax = plt.subplots(figsize=(12,8))
y = m*X + c
ax.plot(X, y, '-r')
ax.scatter(class1_test[[0]], class1_test[[1]], label="Class1")
ax.scatter(class2_test[[0]], class2_test[[1]], label="Class2")
ax.set_ylim([-10, 25])
plt.show()