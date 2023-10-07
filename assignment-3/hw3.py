import os
import sys
import argparse
import jax.numpy as jnp
from jax import grad 
import numpy as np
import matplotlib.pyplot as plt


datapath="./dataset/"


class Optimizer:

	def __init__(self, learning_rate=1., diminish=True):
		self.epoch = 0
		self.lr = learning_rate
		self.diminish_lr = diminish

	def step(self, grad):
		self.epoch += 1
		steplength = self.lr
		if self.diminish_lr:
			steplength /= self.epoch
		return -steplength * grad


class LinearRegressionModel:

	def __init__(self, input_dim, cost="MSE", standardize=True):
		self.w = jnp.ones((input_dim, 1))
		self.loss = getattr(self, cost, self.MSE)
		self.standardize = standardize
		self.optimizer = Optimizer()
		self.grad = grad(self.loss)
		self.weight_history = [self.w]
		self.cost_history = [jnp.nan]
		self.best_epoch = 0

	def train(self, x, y, num_epochs):
		if self.standardize:
			x_mean = x.mean(axis=0)
			x_std = x.std(axis=0)
			x =  (x - x_mean) / x_std
			self.destandardize = self.destandardizer(x_mean, x_std)
		ẋ = jnp.column_stack([jnp.ones_like(x), x])
		if self.optimizer.epoch == 0:
			self.cost_history[0] = self.loss(self.w, ẋ, y)
		last_epoch = self.optimizer.epoch + num_epochs
		while (self.optimizer.epoch < last_epoch):
			self.w += self.optimizer.step(self.grad(self.w, ẋ, y))
			loss = self.loss(self.w, ẋ, y)
			if loss < self.cost_history[self.best_epoch]:
				self.best_epoch = self.optimizer.epoch
			self.weight_history.append(self.w)
			self.cost_history.append(loss)
			yield (self.optimizer.epoch, loss)

	def __call__(self, x):
		w = self.best_w()
		return w[0] + w[1:].T @ jnp.array([x])

	def MSE(self, w, x, y):
		return jnp.mean((x @ w - y) ** 2)

	def MAD(self, w, x, y):
		return jnp.mean(jnp.abs(x @ w - y))
	
	def best_w(self):
		w = self.weight_history[self.best_epoch]
		return self.destandardize(w) if self.standardize else w
	
	def destandardizer(self, m, s):
		def destandardize(w):
			orig_w = w[1:] / s
			orig_b = w[0] - jnp.dot(orig_w , m)
			return jnp.column_stack([orig_b, orig_w]).squeeze()
		return destandardize


def data_plot(title, xlabel, x, ylabel, y, *w_label_pairs):
	_, ax = plt.subplots(figsize=(10,6))
	ax.plot(x, y, linestyle=' ', marker='.')
	x_line = [min(x), max(x)]
	for w, label in w_label_pairs:
		y_line = [w[0] + w[1] * x_val for x_val in x_line]
		ax.plot(x_line, y_line, linestyle='-', marker=None, label=label)
	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.legend()
	plt.savefig(f"image/{title}")


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
	plt.savefig(f"image/{title}")


def np_closed_form_solution(x, y):
	x = jnp.column_stack([jnp.ones_like(x), x])
	optimal_solution = np.linalg.lstsq(x, y, rcond=None)[0].squeeze()
	print("closed form lstsq solution:", optimal_solution)
	return optimal_solution


def print_epoch(epoch, best_epoch, loss):
	print(f"{epoch:<5}{best_epoch:<4}{loss:<12.6f}")


def run_task1():

	file = "student-debt.csv"

	data = np.loadtxt(f"{datapath}{file}", delimiter=",")
	x, y = jnp.array(data[:,:-1]), jnp.array(data[:,-1:])
	len_w = x.shape[1] + 1
	
	print("Training Regression Model using Mean Squared Error")
	mse_model = LinearRegressionModel(len_w, cost="MSE")
	for epoch, loss in mse_model.train(x, y, 100):
		print_epoch(epoch, mse_model.best_epoch, loss)


	print("Training Regression Model using Mean Absolute Deviation")
	mad_model = LinearRegressionModel(len_w, cost="MAD")
	for epoch, loss in mad_model.train(x, y, 100):
		print_epoch(epoch, mad_model.best_epoch, loss)

	
	mse_w = mse_model.best_w()
	mad_w = mad_model.best_w()
	cfs_w = np_closed_form_solution(x, y)

	print("MSE-Model Prediction for x = 2030:", mse_model(2030))

	data_plot("Task 1: Fitted-Line",
		"Year", x,
		"Total Student Debt (Trillion $)", y,
		(cfs_w, f"y={round(cfs_w[1].item(), 3)}x {round(cfs_w[0].item(), 3)}   Closed-Form LstSq"),
		(mad_w, f"y={round(mad_w[1].item(), 3)}x {round(mad_w[0].item(), 3)}   MAD"),
		(mse_w, f"y={round(mse_w[1].item(), 3)}x {round(mse_w[0].item(), 3)}   MSE")
	)

	error_plot("Task 1 Loss",
		(mse_model.cost_history, "MSE   Standardized-Input   LR:1/k   Best_Epoch:1"),
		(mad_model.cost_history, "MAD   Standardized-Input   LR:1/k   Best_Epoch:100")
	)


def run_task2():

	file = "regression-outliers.csv"

	data = np.loadtxt(f"{datapath}{file}",delimiter = ',')
	x, y = jnp.array(data[0,:][np.newaxis].T), jnp.array(data[1,:][np.newaxis].T)
	len_w = x.shape[1] + 1

	print("Training Regression Model using Mean Squared Error")
	mse_model = LinearRegressionModel(len_w, cost="MSE")
	for epoch, loss in mse_model.train(x, y, 100):
		print_epoch(epoch, mse_model.best_epoch, loss)


	print("Training Regression Model using Mean Absolute Deviation")
	mad_model = LinearRegressionModel(len_w, cost="MAD")
	for epoch, loss in mad_model.train(x, y, 100):
		print_epoch(epoch, mad_model.best_epoch, loss)

	mse_w = mse_model.best_w()
	mad_w = mad_model.best_w()
	cfs_w = np_closed_form_solution(x, y)

	data_plot("Task 2: Two Fitted Lines",
		"", x,
		"", y,
		(cfs_w, f"y={round(cfs_w[1].item(), 3)}x+{round(cfs_w[0].item(), 3)}   Closed-Form LstSq"),
		(mad_w, f"y={round(mad_w[1].item(), 3)}x {round(mad_w[0].item(), 3)}   MAD"),
		(mse_w, f"y={round(mse_w[1].item(), 3)}x+{round(mse_w[0].item(), 3)}   MSE")
	)

	error_plot("Task 2 Loss",
		(mse_model.cost_history, "MSE   Standardized-Input   LR:1/k   Best_Epoch:2"),
		(mad_model.cost_history, "MAD   Standardized-Input   LR:1/k   Best Epoch:99")
	)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Linear Regression Model Task Runner")
	
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