import os
import sys
import argparse
import jax.numpy as jnp
import numpy as np
from jax import grad, random
import matplotlib.pyplot as plt


datapath = "./"


key = random.PRNGKey(0)


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


class MultiClassModel:

	def __init__(self, input_dim, num_classes, cost="softmax", normalize=True, regularize=True):
		self.identifier = cost
		self.num_classes = num_classes
		self.w = random.normal(key, (input_dim, num_classes))
		self.normalize = normalize
		if self.normalize:
			self.normalize_w()
		self.regularize = regularize
		self.λ = 1e-3
		self.loss = getattr(self, cost, self.softmax)
		self.optimizer = Optimizer()
		self.grad = grad(self.loss)
		self.weight_history = [self.w]
		self.cost_history = [jnp.nan]
		self.best_epoch = 0

	def train(self, x, y, num_epochs):
		ẋ = self.add_bias_column(x)
		y1 = self.one_hot_encode(y)
		if self.optimizer.epoch == 0:
			self.cost_history[0] = self.loss(self.w, ẋ, y1)
		for _ in range(num_epochs):
			self.w += self.optimizer.step(self.grad(self.w, ẋ, y1))
			if self.normalize:
				self.normalize_w()
			loss = self.loss(self.w, ẋ, y1)
			if loss < self.cost_history[self.best_epoch]:
				self.best_epoch = self.optimizer.epoch
			self.weight_history.append(self.w)
			self.cost_history.append(loss)
			yield (self.optimizer.epoch, loss)

	def __call__(self, x):
		ẋ = self.add_bias_column(x) # Adjusting for input shape (2, N)
		logits = ẋ @ self.best_w()
		exp_logits = jnp.exp(logits - jnp.max(logits, axis=-1, keepdims=True))
		probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
		return probs

	def softmax(self, w, x, y):
		logits = x @ w
		exp_logits = jnp.exp(logits - jnp.max(logits))
		probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
		loss = -jnp.mean(jnp.sum(y * jnp.log(probs), axis=-1))
		if self.regularize:
			loss += self.λ * jnp.linalg.norm(w[1:,:], 'fro')**2
		return loss

	def best_w(self):
		return self.weight_history[self.best_epoch]

	def normalize_w(self):
		self.w /= jnp.linalg.norm(self.w, axis=0, keepdims=True)

	def add_bias_column(self, x):
		return jnp.column_stack([jnp.ones((x.shape[0], 1)), x])

	def one_hot_encode(self, y):
		return (jnp.arange(self.num_classes) == y.astype(int)).astype(float)

#################### Task 3 ###################

"""
Implementing the multi-class classification with Softmax cost; 
verify the implementation is correct by achiving small misclassification rate. 
"""


# A helper function to plot the original data
def show_dataset(x, y):
  y = y.flatten()
  num_classes = np.size(np.unique(y.flatten()))
  accessible_color_cycle = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
  # initialize figure
  plt.figure()

  # color current class
  for a in range(0, num_classes):
    t = np.argwhere(y == a)
    t = t[:, 0]
    plt.scatter(
      x[0, t],
      x[1, t],
      s=50,
      color=accessible_color_cycle[a],
      edgecolor='k',
      linewidth=1.5,
      label="class:" + str(a))
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(bbox_to_anchor=(1.1, 1.05))

  plt.savefig("data.png")
  plt.close()


def show_dataset_labels(x, y, model, n_axis_pts=120):
	y = y.flatten()
	num_classes = np.size(np.unique(y.flatten()))
	accessible_color_cycle = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

	plt.figure()

	axis_range = np.linspace(0.05, 0.95, num=n_axis_pts)
	x1, x2 = np.meshgrid(axis_range, axis_range)
	points_region = np.c_[x1.ravel(), x2.ravel()]
	ŷ_region = model(points_region).argmax(axis=1)

	ŷ_x = model(x).argmax(axis=1)
	correct = jnp.sum(ŷ_x == y)
	accuracy = correct / y.shape[0]

	print(f"final cost: {model.cost_history[model.best_epoch]}")
	print(f"accuracy:  {accuracy:<4}  ({correct}/{y.shape[0]})")

	for a in range(0, num_classes):
		t = np.argwhere(ŷ_region == a)[:, 0]
		plt.scatter(
			points_region[t, 0],
			points_region[t, 1],
			s=5,
			color=accessible_color_cycle[a],
			linewidth=1.5,
			label="class:" + str(a)
		)

	# color current class
	for a in range(0, num_classes):
		t = np.argwhere(y == a)[:, 0]
		plt.scatter(
			x[t, 0],
			x[t, 1],
			s=50,
			color=accessible_color_cycle[a],
			edgecolor='k',
			linewidth=1.5,
			label="class:" + str(a)
		)
		plt.xlabel("x1")
		plt.ylabel("x2")
	plt.legend(bbox_to_anchor=(1.1, 1.05))
	plt.savefig("classifier_label_regions.png")
	plt.close()


def print_epoch(epoch, best_epoch, loss):
	print(f"{epoch:<5}{best_epoch:<5}{loss:<12.6f}".rstrip())


def run_task3():

	data = np.loadtxt(datapath + '4class_data.csv', delimiter=',')

	x, y = data[:-1, :].T, data[-1:, :].T

	model = MultiClassModel(3, 4)

	for epoch, loss in model.train(x, y, 1000):
		print_epoch(epoch, model.best_epoch, loss)

	show_dataset_labels(x, y, model)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(
        description="Multi-class Classification Model Task Runner"
    )
	
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

	run_task3()

	if args.output != sys.stdout:

		args.output.close()