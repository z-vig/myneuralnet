import numpy as np

"""
This is where we define all applicable loss functions for the neural network
"""

def mse(result:'np.ndarray',label:'np.ndarray'):
    diff = result-label
    return (1/len(result))*np.dot(diff,diff)