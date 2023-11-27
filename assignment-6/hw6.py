import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

np.random.seed(0)

datapath = "./"

#################### Task 1 ################### 

def run_task1():

	x = np.loadtxt(datapath+'2d_span_data.csv', delimiter=',').T # (50, 2)

	# PCA Algorithm:

	# 1. Center the data.
	x -= x.mean(axis=0)

	# 2. Create the correlation matrix (λ for numerical stability).
	λ = 10e-7 
	c = (x.T @ x) / x.shape[0] + λ * np.eye(x.shape[1])

	# 3. Compute eigenvalues / vectors of correlation matrix.
	vals, vecs = np.linalg.eigh(c)

	# 4. Extract Principal Components in descending order.
	pc = vecs[vals.argsort()[::-1],:]
	
	# Plots
	_, ax = plt.subplots(1, 2)

	# Plot Original Data
	ax[0].set_title('original data')
	ax[0].set_aspect('equal')
	ax[0].set_xlabel('$X_1$')
	ax[0].set_xlim([-7, 7])
	ax[0].set_ylabel('$X_2$', rotation=0)
	ax[0].set_ylim([-9, 9])
	ax[0].axhline(color='black')
	ax[0].axvline(color='black')
	ax[0].scatter(x[:,0], x[:,1], facecolors='black', edgecolors='white', label='centered data')
	ax[0].arrow(0, 0, *pc[0], linewidth=3, head_width=0.5, head_length=0.5, fc='red', ec='red', zorder=2)
	ax[0].arrow(0, 0, *pc[1], linewidth=3, head_width=0.5, head_length=0.5, fc='red', ec='red', zorder=2)

	encoded_x = x @ vecs
	basis_v = vecs @ vecs.T

	assert np.allclose(basis_v, np.eye(2,2)), "Something is off, v isn't orthogonal."

	# Plot Encoded Data
	ax[1].set_title('encoded data')
	ax[1].set_aspect('equal')
	ax[1].set_xlabel('$C_1$')
	ax[1].set_xlim([-3, 3])
	ax[1].set_ylabel('$C_2$', rotation=0)
	ax[1].set_ylim([-11, 11])
	ax[1].axhline(color='black')
	ax[1].axvline(color='black')
	ax[1].scatter(encoded_x[:,0], encoded_x[:,1], facecolors='black', edgecolors='white', label='centered data')
	ax[1].arrow(0, 0, *basis_v[0], linewidth=3, head_width=0.5, head_length=0.5, fc='red', ec='red', zorder=2)
	ax[1].arrow(0, 0, *basis_v[1], linewidth=3, head_width=0.5, head_length=0.5, fc='red', ec='red', zorder=2)
	
	plt.savefig('task-1-plot')

#################### Task 2 ###################

def run_task2():

	data, _ = datasets.make_blobs \
	(
		n_samples=50,
		centers=3,
		random_state=10
	)

	scree = []

	for k in range(1, 11):

		centroids = np.random.randn(k, data.shape[1])

		assignments = None
		prev_assignments = None

		for t in range(20):

			assignments = np.linalg.norm(data[:,np.newaxis] - centroids, axis=2).argmin(axis=1)
			
			if prev_assignments is not None and np.allclose(assignments, prev_assignments):
				print(f'k={k} converged in {t+1} iterations')
				scree.append(0)
				for i in range(k):
					indices_of_set_i = np.where(assignments==i)[0]
					if indices_of_set_i.size == 0:
						continue
					set_i = data[indices_of_set_i]
					centroids[i] = np.mean(set_i, axis=0)
					scree[-1] += np.mean((set_i - centroids[i])**2)
				break

			prev_assignments = assignments

			for i in range(k):
				indices_of_set_i = np.where(assignments==i)[0]
				if indices_of_set_i.size == 0:
					continue
				set_i = data[indices_of_set_i]
				centroids[i] = np.mean(set_i, axis=0)


		# Produce K=3 plot.
		if (k == 3):
			cluster1 = data[np.where(assignments==0)]
			cluster2 = data[np.where(assignments==1)]
			cluster3 = data[np.where(assignments==2)]
			
			plt.scatter(cluster1[:, 0], cluster1[:, 1], s=50, c='red', alpha=0.5, label='cluster 1')
			plt.scatter(cluster2[:, 0], cluster2[:, 1], s=50, c='blue', alpha=0.5, label='cluster 2')
			plt.scatter(cluster3[:, 0], cluster3[:, 1], s=50, c='green', alpha=0.5, label='cluster 3')

			plt.scatter(centroids[0,0], centroids[0,1], s=150, c='red', alpha=1.0, label='centroid 1', marker='*', ec='white')
			plt.scatter(centroids[1,0], centroids[1,1], s=150, c='blue', alpha=1.0, label='centroid 2', marker='*', ec='white')
			plt.scatter(centroids[2,0], centroids[2,1], s=150, c='green', alpha=1.0, label='centroid 3', marker='*', ec='white')

			plt.xlabel('Feature 1')
			plt.ylabel('Feature 2')
			plt.title('K-means Clustering: Data Points and Centroids')
			plt.legend()
			plt.savefig(f'{k}-means')

	_, ax = plt.subplots()
	plt.title('Scree Plot')
	plt.xticks(range(1,11))
	ax.plot(range(1, 11), scree, label='Sum of Squared Errors')
	ax.set_xlabel('Number of Clusters (k)')
	ax.set_ylabel('Error')
	ax.legend()
	plt.savefig('scree')



if __name__ == '__main__':
	run_task1()
	run_task2()