import jax.numpy as jnp 
from jax import grad 
import numpy as np 
import pandas as pd 
from sklearn.datasets import fetch_openml

import matplotlib.pyplot as plt 

datapath = "./"

np.random.seed(0)

#################### Task 1 ###################

###### Helper functions for task 1 ########
# multi-class linear classification model 
def model(x, w): 
    """
    input: 
    - x: shape (N, P)  
    - W: shape (N+1, C) 

    output: 
    - prediction: shape (C, P) 
    """
    # option 1: stack 1 
    f = x   
    # print("before stack 1, x.shape: ", f.shape)

    # tack a 1 onto the top of each input point all at once
    o = jnp.ones((1, np.shape(f)[1]))
    f = jnp.vstack((o,f))

    # print("after stack 1, the X.shape:", f.shape)

    # compute linear combination and return
    a = jnp.dot(f.T,w)

    # option 2: 
    # a = w[0, :] + jnp.dot(x.T, w[1:, :])
    return a.T


# multi-class softmax cost function 
def multiclass_softmax(w, x_p, y_p):     
    """
    Args:
        - w: parameters. shape (N+1, C), C= the number of classes
        - x_p: input. shape (N, P) 
        - y_p: label. shape (1, P)
    Return: 
        - softmax cost: shape (1,)
    """
    
    # pre-compute predictions on all points
    all_evals = model(x_p, w)
    # print(f"all_evals[:, 0:5].T={all_evals[:, 0:5].T}")

    # logsumexp trick
    maxes = jnp.max(all_evals, axis=0)
    a = maxes + jnp.log(jnp.sum(jnp.exp(all_evals - maxes), axis=0))

    # compute cost in compact form using numpy broadcasting
    b = all_evals[y_p.astype(int).flatten(), jnp.arange(np.size(y_p))]
    cost = jnp.sum(a - b)

    # return average
    return cost/float(np.size(y_p))


def standard_normalize(x):
    std = x.std(axis=0)
    std[std == 0] = 1.0 # Set 0's to 1 to avoid zerodiv
    return (x - x.mean(axis=0)) / std


def pca_sphere(x):
    x_centered = x - x.mean(axis=0)
    λ, ϵ = 1e-7, 1e-5 # Constants for numerical stability 
    c = (x_centered.T @ x_centered) / x.shape[0] + λ * np.eye(x.shape[1])
    vals, vecs = np.linalg.eigh(c)
    encoded_x = x_centered @ vecs
    return encoded_x / np.sqrt(vals + ϵ)[np.newaxis,:]


def gradient_descent(x, y, max_iterations, learning_rate):
    gradient = grad(multiclass_softmax)
    cost_history = []
    accuracy_history = []

    N = x.shape[0]
    C = len(np.unique(y))

    w = np.random.randn(N+1, C)
    ŷ = model(x, w).argmax(axis=0)[np.newaxis,:].T
    cost_history.append(multiclass_softmax(w, x, y))
    accuracy_history.append(np.sum(ŷ==y)/y.shape[0])

    for i in range(max_iterations):
        w -= learning_rate * gradient(w, x, y)
        ŷ = model(x, w).argmax(axis=0)[np.newaxis,:].T
        cost_history.append(multiclass_softmax(w, x, y))
        accuracy_history.append(np.sum(ŷ==y)/y.shape[0])

    return cost_history, accuracy_history


def plot(title, *history_label_pairs, xlabel='Epoch (k)', ylabel='Accuracy'):
    _, ax = plt.subplots(figsize=(10, 6))
    i = range(0, len(history_label_pairs[0][0]))
    if i.stop < 30:
        plt.xticks(i)
    for history, label in history_label_pairs:
        ax.plot(i, history, '-', label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.savefig(title)


def run_task1():
    """
    Produces 5 plots
    """

    ## Uncomment to import MNIST dataset and save it locally. (also convert pandas DF to NumPy arrays)
    x, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    x = x.to_numpy()
    y = y.to_numpy()
    # np.savez('mnist_dataset.npz', X=x, y=y)

    ## Uncomment when the dataset has already been downloaded.
    # data = np.load('mnist_dataset.npz', allow_pickle=True)
    # x = data['X']
    # y = data['y']
    
    y = np.array([float(v) for v in y])[np.newaxis,:].T

    # Sample the 50000 MNIST data points.
    x = x[:50000]
    y = y[:50000]

    optimal_runs = []

    ## Produce Cost History Analysis Plots

    for n in ['original', 'standard normalized', 'pca normalized']:

        ẋ = x

        if n == 'standard normalized':
            ẋ = standard_normalize(x)
        elif n == 'pca normalized':
            ẋ = pca_sphere(x)

        hist_labels = []

        for γ in [-3, -2, -1, 0, +1, +2, +3]:
            print(f"{n} γ={γ}")
            cost, acc = gradient_descent(ẋ.T, y, 10, 10**γ)

            if n == 'original' and γ == -2:
                optimal_runs.append((cost, acc))
            if n == 'standard normalized' and γ == 1:
                optimal_runs.append((cost, acc))
            if n == 'pca normalized' and γ == 2:
                optimal_runs.append((cost, acc))

            hist_labels.append((cost, f'γ = {γ}'))

        if (n == 'original'):
            plot(f"Cost-Histories-{n}", *hist_labels[:4], ylabel="Cost")
        if (n == 'standard normalized'):
            plot(f"Cost-Histories-{n}", *hist_labels[:6], ylabel="Cost")
        if (n == 'pca normalized'):
            plot(f"Cost-Histories-{n}", *hist_labels, ylabel="Cost")

    ## Produce Cost and Accuracy Plots with optimal Learning-Rates

    plot\
    (
        "Cost Plot",
        (optimal_runs[2][0], 'PCA-Sphered (LR=100)'),
        (optimal_runs[1][0], 'Standard-Normalized (LR=10)'),
        (optimal_runs[0][0], 'Original Data (LR=.01)')
    )

    plot\
    (
        "Accuracy Plot",
        (optimal_runs[2][1], 'PCA-Normalized (LR=100)'),
        (optimal_runs[1][1], 'Standard-Normalized (LR=10)'),
        (optimal_runs[0][1], 'Original Data (LR=.01)')
    )



#################### Task 2 ###################


## My Optimizer class from HW3
class Optimizer:

    def __init__(self, learning_rate=1., diminish=False):
        self.epoch = 0
        self.lr = learning_rate
        self.diminish_lr = diminish

    def step(self, grad):
        self.epoch += 1
        steplength = self.lr
        if self.diminish_lr:
            steplength /= self.epoch
        return -steplength * grad


## My linear regression implementation from HW3
class LinearRegressionModel:

    def __init__(self, input_dim, cost="MSE", standardize=True, λ=0, LR=1.):
        self.w = jnp.zeros((input_dim, 1)) #jnp.array(np.random.randn(input_dim, 1)) #jnp.array([22.531954, -0.77271235, 0.86519337, -0.03911824, 0.73002046, -1.7282195, 2.9160872, -0.1049947, -2.7752774, 1.6318269, -1.1303469, -1.9713995, 0.88152444, -3.607877]).reshape(input_dim, 1)#jnp.array(np.random.randn(input_dim, 1)) #jnp.ones((input_dim, 1))
        self.loss = getattr(self, cost, self.MSE)
        self.standardize = standardize
        self.optimizer = Optimizer(learning_rate=LR)
        self.grad = grad(self.loss)
        self.λ = λ
        self.weight_history = [self.w]
        self.cost_history = [jnp.nan]
        self.best_epoch = 0

    def train(self, x, y, num_epochs):
        if self.standardize:
            x_mean = x.mean(axis=0)
            x_std = x.std(axis=0)
            x =  (x - x_mean) / x_std
            self.destandardize = self.destandardizer(x_mean, x_std)
        ẋ = jnp.column_stack([jnp.ones(x.shape[0]), x])
        if self.optimizer.epoch == 0:
            self.cost_history[0] = self.loss(self.w, ẋ, y)
        for i in range(num_epochs):
            if i % 1000 == 0:
                self.optimizer.lr /= 10
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
        return jnp.mean((x @ w - y) ** 2) + self.λ * self.L1_norm(w[1:]) / float(y.size)

    def MAD(self, w, x, y):
        return jnp.mean(jnp.abs(x @ w - y)) + self.λ * self.L1_norm(w[1:])
    
    def L1_norm(self, w):
        return jnp.sum(jnp.abs(w))

    def best_w(self):
        return self.weight_history[self.best_epoch]
        return self.destandardize(w) if self.standardize else w
    
    def destandardizer(self, m, s):
        def destandardize(w):
            orig_w = w[1:] / s
            orig_b = w[0] - jnp.dot(orig_w , m)
            return jnp.column_stack([orig_b, orig_w]).squeeze()
        return destandardize


def print_epoch(epoch, best_epoch, loss):
    print(f"{epoch:<5}{best_epoch:<5}{loss:<12.6f}")


def feature_plot(title, feature_touching_weights):

    # Boston Housing dataset feature names
    feature_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, feature_touching_weights)
    plt.ylim([-4, 3])
    plt.xlabel('Features')
    plt.ylabel('Weight')
    plt.title(title)
    plt.xticks(rotation=45) # Rotate feature names for better readability
    plt.savefig(title)



def run_task2(): 
    # load in data
    csvname =  datapath + 'boston_housing.csv'
    data = np.loadtxt(csvname, delimiter = ',')
    x = data[:-1,:].T
    y = data[-1:,:].T

    costs = []

    for λ in [0, 50, 100, 150]:

        model = LinearRegressionModel(14, λ=λ, LR=.1)

        for epoch, loss in model.train(x, y, 500):
            print_epoch(epoch, model.best_epoch, loss)
        
        feature_plot(f"Feature Touching Weights, λ={λ}", model.best_w()[1:].flatten())

        costs.append(model.cost_history)

    plot\
    (
        f"Task-2-Cost-Plot",
        (costs[0], "λ=0"),
        (costs[1], "λ=50"),
        (costs[2], "λ=100"),
        (costs[3], "λ=150")
    )


if __name__ == '__main__':
    run_task1()
    run_task2()