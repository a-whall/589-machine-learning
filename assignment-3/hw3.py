# hw3.py 


import jax.numpy as jnp 
from jax import grad 
# intro of jax library: https://jax.readthedocs.io/en/latest/notebooks/quickstart.html 
import numpy as np 
import matplotlib.pyplot as plt 

datapath="./"

#################### Task 1 ###################

"""
Fit a linear regression model to the student debt data 
All parts marked "TO DO" are for you to construct.
"""


def run_task1(): 
	# import the dataset
	csvname = datapath + 'student_debt_data.csv'
	data = np.loadtxt(csvname,delimiter=',')

	# extract input - for this dataset, these are times
	x = data[:,0]

	# extract output - for this dataset, these are total student debt
	y = data[:,1]

	print(np.shape(x))
	print(np.shape(y))

	# TODO: fit a linear regression model to the data  



#################### Task 2 ###################

"""
Compare the least squares and the least absolute deviation costs 
All parts marked "TO DO" are for you to construct.
"""

def run_task2():
	# load in dataset
	data = np.loadtxt(datapath + 'regression_outliers.csv',delimiter = ',')
	x = data[:-1,:]
	y = data[-1:,:] 

	print(np.shape(x))
	print(np.shape(y))

	# TODO: fit two linear models to the data 



if __name__ == '__main__':
	run_task1()
	run_task2() 
