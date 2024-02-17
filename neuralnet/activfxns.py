import numpy as np

"""
This is where all applicable activation functions for the neural net are stored.
"""

def sigmoid(arr:'np.ndarray') -> 'np.ndarray':
    return 1 / (1 + np.exp(-1 * arr))

def relu(arr:'np.ndarray') -> 'np.ndarray':
    return np.array([0 if i<0 else i for i in arr]) 