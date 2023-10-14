import os
import sys
import argparse
import inspect
from datetime import datetime
import numpy as np
from jax import grad, config, random
import jax.numpy as jnp
import matplotlib.pyplot as plt


config.update("jax_enable_x64", True)


key = random.PRNGKey(0)


datapath = "./"


class Optimizer:

	def __init__(self, schedule=lambda k:1.):
		self.epoch = 0
		if schedule is None:
			self.schedule = lambda k:1.
		else:
			self.schedule = schedule

	def step(self, grad):
		self.epoch += 1
		return -self.schedule(self.epoch) * grad


class LogRegModel:

	def __init__(self, input_dim, cost="softmax", schedule=None, init="normal", standardize_input=False):
		self.identifier = cost
		if isinstance(init, (int, float)):
			self.initializer_constant = init
			self.initializer = self.constant
		else:
			self.initializer = getattr(self, init, self.normal)
		self.identifier += f"   w_init={init}"
		self.w = self.initializer(input_dim)
		self.loss = getattr(self, cost, self.softmax)
		self.standardize_input = standardize_input
		self.optimizer = Optimizer(schedule=schedule)
		self.identifier += f"   {inspect.getsource(self.optimizer.schedule).strip()}"
		self.grad = grad(self.loss)
		self.weight_history = [self.w]
		self.cost_history = [jnp.nan]
		self.best_epoch = 0

	def train(self, x, y, num_epochs):
		if self.standardize_input:
			self.x_mean = x.mean(axis=0)
			self.x_std = x.std(axis=0)
			x = self.standardize(x)
		ẋ = self.add_bias_column(x)
		if self.optimizer.epoch == 0:
			self.cost_history[0] = self.loss(self.w, ẋ, y)
		for _ in range(num_epochs):
			self.w += self.optimizer.step(self.grad(self.w, ẋ, y))
			loss = self.loss(self.w, ẋ, y)
			if loss < self.cost_history[self.best_epoch]:
				self.best_epoch = self.optimizer.epoch
			self.weight_history.append(self.w)
			self.cost_history.append(loss)
			yield (self.optimizer.epoch, loss)

	def __call__(self, x):
		x = jnp.array(x)
		if self.standardize_input:
			x = self.standardize(x)
		w = self.best_w()
		return w[0] + w[1:].T @ x

	def softmax(self, w, x, y):
		return jnp.mean(jnp.log(1 + jnp.exp(-y * x @ w)))

	def perceptron(self, w, x, y):
		return jnp.mean(jnp.maximum(0, -y * (x @ w)))
	
	def best_w(self):
		return self.weight_history[self.best_epoch]

	def add_bias_column(self, x):
		return jnp.column_stack([jnp.ones((x.shape[0], 1)), x])

	def standardize(self, x):
		if hasattr(self, "x_mean") and hasattr(self, "x_std"):
			return (x - x.mean(axis=0)) / x.std(axis=0)
		else:
			raise ValueError("Mean and std are not set. These are updated during training.")
	
	def constant(self, len_w):
		return jnp.ones((len_w, 1)) * self.initializer_constant

	def normal(self, len_w):
		return random.normal(key, (len_w, 1))

	def xavier(self, len_w):
		return jnp.sqrt(1. / len_w) * self.normal(len_w)


def data_plot(title, xlabel, x, ylabel, y, *w_label_pairs):
    _, ax = plt.subplots(figsize=(10,6))
    x_line = np.linspace(min(x), max(x), 400)
    for w, label in w_label_pairs:
        z = w[0] + w[1] * x_line
        y_line = 2 / (1 + np.exp(-z)) - 1
        ax.plot(x_line, y_line, linestyle='-', marker=None, label=label)
    ax.plot(x, y, linestyle=" ", marker=".", label="Original Data")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.savefig(f"{title}{datetime.now().strftime('%d-%H-%M-%S')}")


def error_plot(title, *args):
	_, ax = plt.subplots(figsize=(10, 6))
	i = range(0, len(args[0][0]))
	if i.stop < 30:
		plt.xticks(i)
	for history, label in args:
		ax.plot(i, history, '-', label=label)
	ax.set_title(title)
	ax.set_xlabel('Epoch (k)')
	ax.set_ylabel('Loss')
	ax.legend()
	plt.savefig(f"{title}{datetime.now().strftime('%d-%H-%M-%S')}")


def accuracy(x, y, w, decision_boundary=0.5):
	ẋ = jnp.column_stack([jnp.ones((x.shape[0], 1)), x])
	probs = 1. / (1. + jnp.exp(-(ẋ @ w)))
	ŷ = (probs > decision_boundary).astype(int) * 2 - 1
	return jnp.sum(ŷ == y) / y.shape[0]


def accuracy_plot(title, x, y, *weighthistory_label_pairs):
	_, ax = plt.subplots(figsize=(10, 6))
	i = range(0, len(weighthistory_label_pairs[0][0]))
	if i.stop < 30:
		plt.xticks(i)
	for history, label in weighthistory_label_pairs:
		accuracies = []
		for w in history:
			accuracies.append(accuracy(x,y,w))
		ax.plot(i, accuracies, '-', label=label)
	ax.set_title(title)
	ax.set_xlabel('Epoch (k)')
	ax.set_ylabel('Loss')
	ax.legend()
	plt.savefig(f"{title}{datetime.now().strftime('%d-%H-%M-%S')}")


def print_epoch(epoch, best_epoch, loss):
	print(f"{epoch:<5}{best_epoch:<5}{loss:<12.6f}".rstrip())


def print_statistics(x, y, w, decision_boundary=0.5):
	ẋ = jnp.column_stack([jnp.ones((x.shape[0], 1)), x])
	logits = ẋ @ w
	probs = 1. / (1. + jnp.exp(-logits))
	predictions = (probs > decision_boundary).astype(int) * 2 - 1
	tp = jnp.sum(jnp.logical_and(predictions == 1, y == 1))
	tn = jnp.sum(jnp.logical_and(predictions == -1, y == -1))
	fp = jnp.sum(jnp.logical_and(predictions == 1, y == -1))
	fn = jnp.sum(jnp.logical_and(predictions == -1, y == 1))
	accuracy = (tp + tn) / y.shape[0]
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	print(f"accuracy:  {accuracy:<.4f} ({tp+tn}/{y.shape[0]})")
	print(f"precision: {precision:<.4f}")
	print(f"recall:    {recall:<.4f}")
	print(f"F1-score:  {2*precision*recall/(precision+recall):<.4f}")


#################### Task 1 ###################
# Implementing the linear classification with Softmax cost;
# verify the implementation is correct by achiving zero misclassification.


def run_task1(): 

	csvname = datapath + "2d_classification_data_v1.csv"
	
	data = np.loadtxt(csvname, delimiter=",")

	x, y = data[:-1, :].T, data[-1:, :].T

	len_w = x.shape[1] + 1

	LR=lambda k:1.0
	s_model = LogRegModel(len_w, cost="softmax", init=3, schedule=LR)
	for epoch, loss in s_model.train(x, y, 2000):
		print_epoch(epoch, s_model.best_epoch, loss)
	print_statistics(x, y, s_model.best_w())
	
	data_plot("Task-1: Fitted Tanh Curve",
		"input", x,
		"output", y,
		(s_model.best_w(), "Fitted Tanh Curve")
	)

	error_plot("Task-1 Cost",
		(s_model.cost_history, s_model.identifier)
	)


#################### Task 2 ###################

"""
Compare the efficacy of the Softmax and 
the Perceptron cost functions in terms of the 
minimal number of misclassifications each can 
achieve by proper minimization via gradient descent 
on a breast cancer dataset. 
"""

def run_task2(): 

	csvname = datapath + "breast_cancer_data.csv"

	data = np.loadtxt(csvname, delimiter=",")

	x, y = data[:-1, :].T, data[-1:, :].T

	len_w = x.shape[1] + 1

	iterations = 2000

	LR=lambda k:1/(k%50+1)
	s_model = LogRegModel(len_w, cost="softmax", schedule=LR, init=1)
	for epoch, loss in s_model.train(x, y, iterations):
		print_epoch(epoch, s_model.best_epoch, loss)
	print_statistics(x, y, s_model.best_w())

	LR=lambda k:0.01
	p_model = LogRegModel(len_w, cost="perceptron", schedule=LR, init=1)
	for epoch, loss in p_model.train(x, y, iterations):
		print_epoch(epoch, p_model.best_epoch, loss)
	print_statistics(x, y, p_model.best_w())

	error_plot(f"Task-2 Loss",
		(p_model.cost_history, p_model.identifier),
		(s_model.cost_history, s_model.identifier)
	)

	accuracy_plot(f"Task-2 Accuracy",
		x, y,
		(p_model.weight_history, p_model.identifier),
		(s_model.weight_history, s_model.identifier)
	)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Logistic Regression Model Task Runner")
	
	parser.add_argument(
		'-o', '--output',
		nargs="?",
		const=sys.stdout,
		default=os.devnull,
		type=argparse.FileType('w'),
		help="Control program output behavior. Use -o to provide a filename. If -o is used without a filename, output goes to stdout."
	)

	args = parser.parse_args()

	sys.stdout = args.output

	run_task1()

	run_task2()

	if args.output != sys.stdout:

		args.output.close()
