

import jax.numpy as jnp
import numpy as np  
from jax import grad 
# intro of jax library: https://jax.readthedocs.io/en/latest/notebooks/quickstart.html 


import matplotlib.pyplot as plt 

#################### Task 1 ###################

"""
In this exercise you will implement gradient descent using the hand-computed derivative.
All parts marked "TO DO" are for you to construct.
"""
def cost_func(w):
	"""
	Params: 
	- w (weight)

	Returns: 
	- cost (value of cost function)
	"""
	## TODO: calculate the cost given w
	pass

def gradient_func(w):
	"""
	Params: 
	- w (weight)

	Returns: 
	- grad (gradient of the cost function)
	"""
	## TODO: calculate the gradient given w
	pass

def gradient_descent(g, gradient, alpha,max_its,w):
	"""
	Params: 
	- g (input function), 
	- gradient (gradient function that computes the gradients of the variable)
	- alpha (steplength parameter), 
	- max_its (maximum number of iterations), 
	- w (initialization)

	Returns: 
	- cost_history 
	"""

	# run the gradient descent loop
	cost_history = [g(w)]        # container for corresponding cost function history
	for k in range(1,max_its+1):       
		# TODO: evaluate the gradient, store current weights and cost function value

		



		# collect final weights
		cost_history.append(g(w))  
	return cost_history



def run_task1(): 
	print("run task 1 ...")
	# TODO: Three seperate runs using different steplength 
	

	print("task 1 finished")



#################### Task 2 ###################

"""
In this exercise you will implement gradient descent 
using the automatically computed derivative.
All parts marked "TO DO" are for you to construct.
"""



def gradient_descent_auto(g,alpha,max_its,w, diminishing_alpha=False):
	"""
	
	gradient descent function using automatic differentiator 
	Params: 
	- g (input function), 
	- alpha (steplength parameter), 
	- max_its (maximum number of iterations), 
	- w (initialization)
	
	Returns: 
	- weight_history
	- cost_history

	"""
	# TODO: compute gradient module using jax
	

	# run the gradient descent loop
	weight_history = [w]           # container for weight history
	cost_history = [g(w)]          # container for corresponding cost function history
	for k in range(1, max_its+1):
		# TODO: evaluate the gradient, store current weights and cost function value
		


		# record weight and cost
		weight_history.append(w)
		cost_history.append(g(w))
	return weight_history,cost_history

def run_task2(): 
	print("run task 2 ...")
	# TODO: implement task 2  



	print("task 2 finished")


if __name__ == '__main__':
	run_task1()
	run_task2() 



