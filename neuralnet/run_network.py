from neuralnet import NeuralNet
import activfxns
import lossfxns
import numpy as np
from ml_datasets import mnist
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

"""
Here we define functions for running the neural network using the NeuralNet class
"""

def run_sample(nnobj:'NeuralNet',itrs:'int') -> None:
    return None

def train_network(data:'np.ndarray',labels:'np.ndarray',itrs:'int',nlyrs:'int',lyr_sizes:'list',nout:'int') -> 'np.ndarray':
    print(
        f'Running Neural Network for {itrs} iterations, with an output vector of length {nout}'
    )

    for row in tqdm(range(0,data.shape[0])):
        nnobj = NeuralNet(data[row,:],nlyrs,lyr_sizes,nout,activfxns.sigmoid,lossfxns.mse,training_label=labels[row,:])

        lyr_list = nnobj.forward_prop()
    
    return None



if __name__ == "__main__":
    start = time.time()

    mnist = mnist()
    data = mnist[0][0]
    labels = mnist[0][1]
    sample1 = data[0,:]
    label1 = mnist[0][1][0,:]
    test_nnobj = NeuralNet(sample1,2,[16,16],10,activfxns.sigmoid,lossfxns.mse,training_label = label1)

    train_network(data,labels,100,2,[16,16],10)


    def plot_sample(nnobj:'NeuralNet'):
        plt.figure()
        test_lyr_list = run_sample(test_nnobj,10)
        for n,i in enumerate(test_lyr_list):
            plt.scatter(n*np.ones(len(i)),np.arange(0,len(i)),s=100,c=np.array([[0,0,1,a] for a in i]),edgecolors='k')
            if n < len(test_lyr_list)-1:
                arrow_count = 0
                for y in range(0,len(i)):
                    for dy in range(0,len(test_lyr_list[n+1])):
                        alph = nnobj.weights[1].flatten()[arrow_count]
                        if alph>0:
                            plt.arrow(n,y,1,dy-y,alpha=max(0,alph-.05),color='green')
                        elif alph<0:
                            plt.arrow(n,y,1,dy-y,alpha=max(0,abs(alph)-0.5),color='red')
                        arrow_count += 1
        
    # plot_sample(nnobj)
    # print(nnobj.back_prop())
    end = time.time()

    print(f'Completed in {end-start:.2f} Seconds')

    plt.show()