from neuralnet import NeuralNet
import numpy as np
from ml_datasets import mnist
import time

"""
Here we define functions for running the neural network using the NeuralNet class
"""

def run_network(nnobject:'NeuralNet',itrs:'int') -> 'np.ndarray':
    print(
        f'Running Neural Network for {itrs} iterations, with an output vector of length {nnobject.nout}'
    )

    

if __name__ == "__main__":
    start = time.time()

    data = mnist()[0][0]
    sample1 = data[0,:]
    nnobj = NeuralNet(sample1,2,10)

    run_network(nnobj,10)
    end = time.time()

    print(f'Completed in {end-start:.2f} Seconds')