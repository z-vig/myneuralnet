import numpy as np
from timeit import timeit

class NeuralNet():
    """
    This is a class to store the input of a single sample into a neural network.

    Attributes:
        forward_prop       Runs the forward propagation step
        back_prop          Runs the backward propagation step
    """
    def __init__(self,input_vec:'np.ndarray',nlyrs:'int',lyr_sizes:'list',nout:'int',activfxn:'function',loss_fxn:'function',training_label = False) -> None:
        self.current_lyr = input_vec
        self.nlyrs = nlyrs
        self.lyr_sizes = lyr_sizes
        self.nout = nout

        weight_sizes = [len(input_vec),*lyr_sizes,nout]
        self.weights = [np.random.uniform(-1,1,(weight_sizes[i+1],weight_sizes[i])) for i in range(0,len(weight_sizes)-1)]

        self.biases = [np.random.uniform(0,4,(i)) for i in lyr_sizes]

        self.output = np.empty(shape=(nout),dtype='float')

        self.activfxn = activfxn

        if training_label.all() != False:
            self.label = training_label
        else:
            self.label = [0]*10

        self.loss_fxn = loss_fxn

    def forward_prop(self):
        lyr_list = []
        for lyr_count,weight_array in enumerate(self.weights):

            self.current_lyr = np.matmul(weight_array,self.current_lyr)

            if lyr_count < len(self.biases):
                self.current_lyr -= self.biases[lyr_count]

            self.current_lyr = self.activfxn(self.current_lyr)
            lyr_list.append(self.current_lyr)

        self.output = lyr_list[-1]
        
        return lyr_list
    
    def back_prop(self):
        return self.loss_fxn(self.output,self.label)
        
            
            