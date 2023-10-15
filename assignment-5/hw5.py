import jax.numpy as jnp
import numpy as np
from jax import grad
import matplotlib.pyplot as plt

datapath = "./"

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

def show_dataset_labels(x, y, modelf, n_axis_pts=120):
  y = y.flatten()
  num_classes = np.size(np.unique(y.flatten()))
  accessible_color_cycle = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
  # initialize figure
  plt.figure()

  # fill in label regions using scatter points
  # get (x1, x2) for plot region
  anyax = np.linspace(0.05, 0.95, num=n_axis_pts)
  xx = np.meshgrid(anyax, anyax)
  xx_vars = np.reshape(xx, (2, n_axis_pts **2))
  # get class weights from classifier model
  z = modelf(xx_vars)
  # get class label from model output
  y_hat = z.argmax(axis=1)

  for a in range(0, num_classes):
    t = np.argwhere(y_hat == a)
    t = t[:, 0]
    plt.scatter(
      xx_vars[0, t],
      xx_vars[1, t],
      s=5,
      color=accessible_color_cycle[a],
      linewidth=1.5,
      label="class:" + str(a))

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
  plt.savefig("classifier_label_regions.png")
  plt.close()


def run_task3():
  # load in dataset
  data = np.loadtxt(datapath + '4class_data.csv', delimiter=',')

  # get input/output pairs
  x = data[:-1, :]
  y = data[-1:, :]

  print(np.shape(x))
  print(np.shape(y))

  show_dataset(x, y)

  # show data classified with dummy multiclass model
  def dummy_classifier_model(xs):
    y_hats = np.zeros((np.shape(xs)[1], 4))
    ys = ((1 - xs[0, :]) > xs[1, :]).astype(int)
    ys[np.where(xs[0,:] > xs[1,:])] = ys[np.where(xs[0,:] > xs[1,:])] + 2
    for i, e in enumerate(ys):
      y_hats[i,e] = 1
    return y_hats
  show_dataset_labels(x, y, dummy_classifier_model)

  # TODO: fill in your code


if __name__ == '__main__':
  run_task3()