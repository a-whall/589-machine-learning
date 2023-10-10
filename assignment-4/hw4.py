import numpy as np
from jax import grad
import jax.numpy as jnp 
import matplotlib.pyplot as plt 

datapath = "./"

#################### Task 1 ###################

"""
Implementing the linear classification with Softmax cost; 
verify the implementation is correct by achiving zero misclassification. 
"""

def run_task1(): 
	# load in data
	csvname = datapath + '2d_classification_data_v1.csv'
	data = np.loadtxt(csvname,delimiter = ',')

	# take input/output pairs from data
	x = data[:-1, :]
	y = data[-1:, :] 

	print(np.shape(x)) # (1, 11)
	print(np.shape(y)) # (1, 11)

	# TODO: fill in the rest of the code 



#################### Task 2 ###################

"""
Compare the efficacy of the Softmax and 
the Perceptron cost functions in terms of the 
minimal number of misclassifications each can 
achieve by proper minimization via gradient descent 
on a breast cancer dataset. 
"""

def run_task2(): 
	# data input
	csvname = datapath + 'breast_cancer_data.csv'
	data = np.loadtxt(csvname,delimiter = ',')

	# get input and output of dataset
	x = data[:-1, :]
	y = data[-1:, :] 

	print(np.shape(x)) # (8, 699)
	print(np.shape(y)) # (1, 699)
	
	# TODO: fill in the rest of the code 

if __name__ == '__main__':
	run_task1()
	run_task2()



