import numpy as np
import pandas as pd


'''layers will  be in the format (input_layer,hidden_layer_1,...,hidden_layer_n,output_layer)
   activation will depend on classifier or regressor and will be like ("relu","sigmoid",...,"relu")
   learning rate will be a simple initial learning rate value that will change dyanmically with adam or momentum not sure yet
   momentum -  TODO
   cost_func will again depend on classifier or regressor and then it will be a simple string '''

class ANN_classifier:
    def __init__(self,layers,Activations,learning_rate,momentum,cost_func):
        self.learning_rate = learning_rate
        self.Activations = Activations
        self.layers = layers
        self.momentum = momentum
        self.cost_func = cost_func
        self.bias = {}
        for i in range(1,len(layers)):
            self.bias[i] = np.zeros()

        pass

class ANN_regressor:
    def __init__(self,layers,Activations,learning_rate,momentum,cost_func):
        pass