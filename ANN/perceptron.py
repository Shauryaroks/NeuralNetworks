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
        self.learningRate = 0.1
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
    
    def relu_deriv(self,Z):
        return (Z > 0).astype(float)

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
    
    def error(self, y_pred):
        y_true = self.one_hot()
        loss = - np.sum(y_true * np.log(y_pred+1e-9))
        return loss
    
    def cross_entropy(self):
        pass


    def backprop(self, y_pred, forward_vals):
        y_true = self.one_hot()
        delta = {}
        loss = self.error(y_pred)
        print("loss:", loss)

        delta[f"A{self.layers - 1}"] = (y_pred - y_true) / y_true.shape[0] # only works for softmax + cross entropy loss

        for i in reversed(range(self.layers - 1)):
            Z = forward_vals[f"Z{i+1}"]
            
            if i == self.layers - 2:
                # final layer (softmax), already has correct delta
                delta[f"Z{i+1}"] = delta[f"A{i+1}"]
            else:
                delta[f"Z{i+1}"] = delta[f"A{i+1}"] * self.relu_deriv(Z)
            
            delta[f"A{i}"] = np.dot(delta[f"Z{i+1}"], self.weights[f"w{i}"].T)
            self.weights[f"w{i}"] -= self.learningRate * np.dot(forward_vals[f"A{i}"].T, delta[f"Z{i+1}"])
            self.bias[f"b{i}"] -= self.learningRate * np.sum(delta[f"Z{i+1}"], axis=0, keepdims=True)
                    
    def accuracy(self, y_pred):
        predictions = np.argmax(y_pred, axis=1)
        labels = self.y_train
        return np.mean(predictions == labels)

    def train(self, epochs=1000):
        for epoch in range(epochs):
            forward_vals = self.ForwardProp()
            y_pred = forward_vals["probabilities"]
            self.backprop(y_pred, forward_vals)
            if epoch % 10 == 0:
                acc = self.accuracy(y_pred)
                print(f"Epoch {epoch}, Loss: {self.error(y_pred):.4f}, Accuracy: {acc:.4f}")


        
        '''for layer in range(-1,-1*len(self.structure)-2):
            delta[layer] = []
            for neuron in range(self.structure(layer)):
                '''
                
            
                

                



