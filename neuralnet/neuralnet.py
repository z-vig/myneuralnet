import numpy as np
from timeit import timeit

class NeuralNet():
    """
    This is a class to store the input of a single sample into a neural network.

    Attributes:
        forward_prop       Runs the forward propagation step
        back_prop          Runs the backward propagation step
    """
    def __init__(self,input_vec:'np.ndarray',nlyrs:'int',nout:'int') -> None:
        self.input = input_vec
        self.nlyrs = nlyrs
        self.nout = nout
        self.weights = []
        self.biases = []
        self.output = np.empty(shape=(nout),dtype='float')


    def forward_prop(self):
        print(self.input_vector.shape)
        print(self.nlyrs)