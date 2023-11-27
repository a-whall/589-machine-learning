import inspect
import numpy as np
import pandas as pd
from jax import grad, random
import jax.numpy as jnp
import matplotlib.pyplot as plt
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm
np.random.seed(0)



datapath = "./"
key = random.PRNGKey(0)



class Optimizer:

    def __init__(self, learning_rate):
        self.i = 0
        self.learning_rate = learning_rate

    def step(self, grad):
        self.i += 1
        return -self.learning_rate * grad



class MultiClassModel:

    def __init__(self, N, C, loss="softmax", data_normalization=None, regularization=None, normalize_weights=False, optimizer=None):
        """
        Args:
            N: Number of feature touching weights, w ∊ (N+1 x C).
            C: Number of classes in the classification problem.
            loss: Member function to minimize during gradient descent.
            data_normalization: Type of normalization to apply on input data. [eg {None|"std"|"pca"}]
            regularization: Tuple of (Regularization type, Regularization strength) [eg {None|({"L1"|"L2"}, λ)}]
            normalize_weights: Whether to normalize the weights after each step of gradient descent.
        """
        self.identifier = loss

        self.C = C

        self.W = random.normal(key, (N+1, C))

        self._validate_loss_param(loss)
        self.loss = getattr(self, loss)

        self.normalize_data = data_normalization is not None
        if self.normalize_data:
            self._validate_data_normalization_param(data_normalization)
            self.data_normalizer = getattr(self, data_normalization)

        self.regularize = regularization is not None and regularization[0] is not None
        if self.regularize:
            self._validate_regularization_params(regularization)
            reg_type, reg_strength = regularization
            self.regularizer = getattr(self, reg_type)
            self.λ = reg_strength
        
        self.normalize_weights = normalize_weights
        if self.normalize_weights:
            self.normalize_w()

        self.optimizer = Optimizer() if optimizer is None else optimizer
        self.grad = grad(self.loss)
        self.weight_history = [self.W]
        self.cost_history = [jnp.nan]
        self.accuracy_history = []
        self.best_i = 0

    def train(self, x, y, max_iterations):
        """
        Args:
            x (P x N): inputs
            y (P x 1): labels
        """
        if self.normalize_data:
            x = self.data_normalizer(x)
        ẋ = self.add_bias_column(x)
        y1 = self.one_hot_encode(y)
        if self.optimizer.i == 0:
            self.cost_history[0] = self.loss(self.W, ẋ, y1)
            self.accuracy(ẋ, y)
            yield (0, 0, self.cost_history[0])
        for _ in range(max_iterations):
            self.W += self.optimizer.step(self.grad(self.W, ẋ, y1))
            if self.normalize_weights:
                self.normalize_w()
            loss = self.loss(self.W, ẋ, y1)
            if loss < self.cost_history[self.best_i]:
                self.best_i = self.optimizer.i
            self.accuracy(ẋ, y)
            self.weight_history.append(self.W)
            self.cost_history.append(loss)
            if self.optimizer.i % 20 == 0:
                if np.mean(self.cost_history[-20:-10]) - np.mean(self.cost_history[-10:]) < 1e-8:
                    if self.optimizer.learning_rate > 1e-3:
                        self.optimizer.learning_rate *= 0.1
                        #print(f"reducing learning rate to {self.optimizer.learning_rate}")
                    else:
                        #print(f"convergence detected at {self.optimizer.i}")
                        break
            yield (self.optimizer.i, self.best_i, loss)

    def __call__(self, x):
        logits = self.add_bias_column(x) @ self.best_W()
        exp_logits = jnp.exp(logits - jnp.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        return probs

    def softmax(self, w, x, y):
        logits = x @ w
        max_logits = jnp.max(logits, axis=1, keepdims=True)
        logsumexp = max_logits + jnp.log(jnp.exp(logits - max_logits).sum(axis=1, keepdims=True))
        probs = jnp.exp(logits - logsumexp)
        loss = -jnp.mean(jnp.sum(y * jnp.log(probs + 1e-9), axis=-1))
        if self.regularize:
            loss += self.λ * self.regularizer(w)
        return loss

    def accuracy(self, x, y, use_best_w=False, return_result=False):
        W = self.best_W() if use_best_w else self.W
        logits = x @ W
        ŷ = jnp.argmax(logits, axis=1)
        if return_result:
            return jnp.mean(ŷ == y.flatten())
        self.accuracy_history.append(jnp.mean(ŷ == y.flatten()))

    def best_W(self):
        return self.weight_history[self.best_i]

    def normalize_w(self):
        self.W /= jnp.linalg.norm(self.W, axis=0, keepdims=True)

    def add_bias_column(self, x):
        return jnp.column_stack([jnp.ones((x.shape[0], 1)), x])

    def one_hot_encode(self, y):
        return (jnp.arange(self.C) == y.astype(int)).astype(float)

    def std(self, data):
        return (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-9)

    def pca(self, data):
        data -= data.mean(axis=0)
        cov = (data.T @ data) / data.shape[0] + 1e-7 * np.eye(data.shape[1])
        vecs = np.linalg.eigh(cov)[1]
        sphered_data = data @ vecs
        assert np.allclose(vecs @ vecs.T, np.eye(vecs.shape[0])), "not orthogonal"
        return sphered_data

    def l1(self, w):
        return jnp.sum(jnp.abs(w[1:, :]))

    def l2(self, w):
        return jnp.linalg.norm(w[1:, :], 'fro')**2

    def _validate_loss_param(self, loss):
        if not isinstance(loss, str):
            raise ValueError(f"Invalid loss specifier. Expected str, got {type(loss).__name__} instead.")
        if not hasattr(self, loss):
            raise ValueError(f"Invalid loss. {type(self).__name__} has no attribute {loss}.")
        if loss != "softmax" and loss != "perceptron":
            raise ValueError(f"Invalid loss. Expected a loss function: 'softmax' or 'perceptron'.")

    def _validate_data_normalization_param(self, data_normalization):
        if not isinstance(data_normalization, str):
            raise ValueError(f"Invalid data_normalization parameter. Expected a str, got {type(data_normalization).__name__} instead.")
        if not hasattr(self, data_normalization):
            raise ValueError(f"Invalid data_normalization parameter. {type(self).__name__} has no attribute {data_normalization}.")
        if data_normalization != "std" and data_normalization != "pca":
            raise ValueError(f"Invalid data_normalization parameter. Expected a normalization technique abbreviation. Use 'std' to apply standard normalization, and 'pca' to apply pca sphering.")

    def _validate_regularization_params(self, regularization):
        if not (isinstance(regularization, tuple) and len(regularization) == 2):
            raise ValueError("Invalid regularization parameter. Expected a tuple (regularization type: str, regularization strength: float).")
        if not isinstance(regularization[0], str):
            raise ValueError(f"Regularization type must be a string, got {type(regularization[0]).__name__} instead.")
        if regularization[0] != "l1" and regularization[0] != "l2":
            raise ValueError(f"Invalid regularization type: '{regularization[0]}'. Expected a valid method name: 'l1' or 'l2'")
        if not isinstance(regularization[1], (float, int)):
            raise ValueError(f"Regularization strength must be a numeric value, got {type(regularization[1]).__name__} instead.")



def print_epoch(epoch, best_epoch, loss):
    print(f"{epoch:<5}{best_epoch:<5}{loss:<12.6f}".rstrip())



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



def accuracy_plot(title, *acchistory_label_pairs):
    _, ax = plt.subplots(figsize=(10, 6))
    i = range(0, len(acchistory_label_pairs[0][0]))
    if i.stop < 30:
        plt.xticks(i)
    for history, label in acchistory_label_pairs:
        ax.plot(i, history, '-', label=label)
    ax.set_title(title)
    ax.set_xlabel('Epoch (k)')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig(f"{title}{datetime.now().strftime('%d-%H-%M-%S')}")



def train_test_split(*arrays, test_size=0.2, shuffle=True, rand_seed=0):
    """
    Note:
        Array shape expected: (P x N)
    """
    np.random.seed(rand_seed)
    array_len = len(arrays[0])
    split_idx = int(array_len * (1 - test_size))
    indices = np.arange(array_len)
    if shuffle:
        np.random.shuffle(indices)
    result = []
    for array in arrays:
        if shuffle:
            array = array[indices, :]
        train = array[:split_idx, :]
        test = array[split_idx:, :]
        result.extend([train, test])
    return result



def kfold(X, y, lr, norm, reg_type, reg_strength, k=5):
    P = X.shape[0]
    indices = np.arange(P)
    np.random.shuffle(indices)
    folds, current = [], 0
    for fold_size in [P // k + (i < P % k) for i in range(k)]:
        next = current + fold_size
        folds.append(indices[current:next])
        current = next
    accuracies = []
    for i in range(k):
        optimizer = Optimizer(learning_rate=lr)
        model = MultiClassModel(X.shape[1], 2, data_normalization=norm, regularization=(reg_type, reg_strength), optimizer=optimizer)
        test_idx = folds[i]
        train_idx = np.setdiff1d(indices, test_idx)
        train_X, test_X = X[train_idx], X[test_idx]
        train_y, test_y = y[train_idx], y[test_idx]
        for _ in model.train(train_X, train_y, max_iterations=1000):
            pass
        accuracy = model.accuracy(model.add_bias_column(test_X), test_y, return_result=True)
        accuracies.append(accuracy)
    return np.mean(accuracies)

csvname = datapath + 'new_gene_data.csv'
data = np.loadtxt(csvname, delimiter=',')
x = data[:-1, :].T # (72, 7128)
y = data[-1:, :].T # (72, 1)
y = (y+1)/2 # remap for multiclass solver.

learning_rates = np.linspace(0.001, 1, 11)
reg_strengths = np.linspace(0.001, 1, 11)

def eval_lr(i):
    return [kfold(x,y,learning_rates[i],"std","l1",rs) for rs in reg_strengths]

def run_task1():

    print(reg_strengths)

    with Pool(processes=11) as p:
        r = len(learning_rates)
        avg_v_acc = list(tqdm(p.imap(eval_lr, range(r)), total=r))

    print(avg_v_acc)
    np.savetxt(datapath + f"avg_v_acc_{datetime.now().strftime('%d-%H-%M-%S')}", avg_v_acc)

    # for i, lr in enumerate(learning_rates):
    #     for j, rs in enumerate(reg_strengths):
    #         avg_v_acc[i,j] = kfold(x, y, lr, "std", "l1", rs)
    #         print(i,j)

    hparam1_grid, hparam2_grid = np.meshgrid(learning_rates, reg_strengths)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(hparam1_grid, hparam2_grid, avg_v_acc, cmap='viridis')

    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Regularization Strength')
    ax.set_zlabel('Average Validation Accuracy')
    ax.set_title('Hyperparameter Optimization Space (K-Fold Cross-Validation)')
    plt.show()



if __name__ == '__main__':
    run_task1()