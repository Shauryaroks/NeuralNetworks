import numpy as np
import random
import math


class ANN:
    '''here y_train is simply like straightforward outputs and X_train is [[first data][second data point]...number of Xtrain]'''
    def __init__(self,y_train,X_train,structure = (784,16,16,10),seed = None):
        self.structure = structure
        self.layers = len(structure)
        self.y_train = y_train
        self.X_train = X_train
        self.weights = {}
        self.bias = {}
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        for i in range(len(structure)-1):
            self.bias[f"b{i}"] = np.random.randn(1,structure[i+1])*0.01
            self.weights[f"w{i}"] = np.random.randn(structure[i], structure[i+1]) * np.sqrt(2 / structure[i])

    def ForwardProp(self):
        values = {}
        values["A0"] = self.X_train 
        for i in range(self.layers-1):
            Z = np.matmul(values[f"A{i}"], self.weights[f"w{i}"]) + self.bias[f"b{i}"]
            values[f"Z{i+1}"] = Z
            if i<self.layers-2:
                values[f"A{i+1}"] = self.activate(Z)   #here relu is being used
            else: #this will vary for multiclassification or binary classification usecase
                values["probabilities"] = self.softmax(Z) 
        return values
    
    def activate(self,matrix):
        relu = lambda x: np.maximum(0, x)
        return relu(matrix)

    def softmax(self, matrix):
        # subtract max for numerical stability (along axis=1 for row-wise)
        shifted = matrix - np.max(matrix, axis=1, keepdims=True) #understand why TODO
        exp_values = np.exp(shifted)
        softmax_output = exp_values / np.sum(exp_values, axis=1,keepdims=True)
        return softmax_output
    
    def one_hot(self, cats=[0,1,2,3,4,5,6,7,8,9]):
        # Create a mapping from category to index
        cat_to_idx = {cat: idx for idx, cat in enumerate(cats)}
        # Initialize one-hot encoded matrix
        num_samples = len(self.y_train)
        num_classes = len(cats)
        one_hot_encoded = np.zeros((num_samples, num_classes))
        # Fill in the one-hot encoding
        for i, label in enumerate(self.y_train):
            if label in cat_to_idx:
                one_hot_encoded[i, cat_to_idx[label]] = 1
        
        return one_hot_encoded
    
    def cross_entropy_loss(self, y_pred):
        y_true = self.one_hot()
        loss = - np.sum(y_true * np.log(y_pred+1e-9))/len(y_true)
        return loss
    
    def backprop(self,y_pred):
        y_true = self.one_hot()
        delta = {}
        for layer in range(-1,-1*len(self.layers)-2):
            delta[layer] = []
            for neuron in range(self.layers(layer)):
                delta[layer].append()
            
                

                



