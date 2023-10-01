import jax.numpy as jnp
import numpy as np
from jax import grad # intro of jax library: https://jax.readthedocs.io/en/latest/notebooks/quickstart.html
import matplotlib.pyplot as plt


#################### Task 1 ###################
# In this exercise you will implement gradient descent using the hand-computed derivative.
# All parts marked "TO DO" are for you to construct.


def cost_func(w):
	"""
	Params:
	- w (weight)

	Returns:
	- cost (value of cost function)
	"""

	return 0.02 * (w ** 4 + w ** 2 + 10 * w)


def gradient_func(w):
	"""
	Params:
	- w (weight)

	Returns:
	- grad (gradient of the cost function)
	"""

	return 0.08 * w ** 3 + 0.04 * w + 0.2


def gradient_descent(g, gradient, alpha, max_its, w):
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

	cost_history = [g(w)]

	for k in range(1, max_its + 1):

		w = w - alpha * gradient(w)

		cost_history.append(g(w))

	return cost_history


def alpha_plot(filename, *alphas):

	_, ax = plt.subplots()

	# plt.xticks(range(0, 21)) # needed to make task 2 plot x-axis integers -_-

	for data, alpha in alphas:

		ax.plot(range(0, len(data)), data, '-', label=f'alpha = {alpha}')

	ax.set_title('GD Cost History')

	ax.set_xlabel('Iteration')

	ax.set_ylabel('Cost')

	ax.legend()

	plt.savefig(filename)


def run_task1(): 
	
	print("run task 1 ...")

	cost1 = gradient_descent(cost_func, gradient_func, alpha=1, max_its=1000, w=2)
	
	cost2 = gradient_descent(cost_func, gradient_func, alpha=0.1, max_its=1000, w=2)
	
	cost3 = gradient_descent(cost_func, gradient_func, alpha=0.01, max_its=1000, w=2)

	alpha_plot('task-1', (cost1, 1.0), (cost2, 0.1), (cost3, 0.01))

	print("task 1 finished")


#################### Task 2 ###################
# In this exercise you will implement gradient descent 
# using the automatically computed derivative.
# All parts marked "TO DO" are for you to construct.


def gradient_descent_auto(g, alpha, max_its, w, diminishing_alpha=False):
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
	
	gradient_func = grad(g)
	
	a = alpha

	weight_history = [w]

	cost_history = [g(w)]

	for k in range(1, max_its+1):
		
		if diminishing_alpha:
			a = alpha / k
		
		w = w - a * gradient_func(w)

		weight_history.append(w)

		cost_history.append(g(w))

	return weight_history,cost_history


def run_task2(): 
	
	print("run task 2 ...")

	w_fixed, c_fixed = gradient_descent_auto(jnp.abs, 0.5, 20, 2.0, False)

	w_dimin, c_dimin = gradient_descent_auto(jnp.abs, 1.0, 20, 2.0, True)

	alpha_plot('task-2', (c_fixed, 0.5), (c_dimin, '1/k'))

	print("task 2 finished")


if __name__ == '__main__':

	run_task1()

	run_task2()