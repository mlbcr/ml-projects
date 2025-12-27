import random
import numpy as np

class KMeans:
    def __init__(self, n_clusters=2, random_state=42, max_iters=100):
        self.n_clusters = n_clusters
        self.centroids = None
        self.max_iters = max_iters
        random.seed(random_state)

    def _find_closest(self, X, centroids):
        idx = np.zeros(X.shape[0], dtype=int)

        for i in range(X.shape[0]):
            distance = []
            for j in range(centroids.shape[0]):
                norm_ij = np.linalg.norm(X[i] - centroids[j])
                distance.append(norm_ij)

            idx[i] = distance.index(min(distance))
        return idx
    
    def _centroid_mean(self, X, idx, K):
        m, n = X.shape
        centroids = np.zeros((K, n))
        
        for k in range(K):
            points = X[idx == k]
            centroids[k] = np.mean(points, axis=0)
        
        return centroids
    
    def fit(self, X):
        m, n = X.shape

        # Generate random centroids
        centroids = []

        for i in range(self.n_clusters):
            c = []
            for col in range(n):
                c.append(random.uniform(min(X[:, col]), max(X[:, col])))
            centroids.append(c)

        centroids = np.array(centroids)
        self.centroids = centroids


        # Find closest centroids
        idx = self._find_closest(X, centroids)
        print("Initial clusters: ", idx)
        print("K-Means iteration 0/%d" % (self.max_iters))

        for i in range(1, self.max_iters):
            print(f"Iteration {i}/{self.max_iters-1}")
            
            idx = self._find_closest(X, centroids)

            centroids = self._centroid_mean(X, idx, self.n_clusters)
    
        return self.centroids, idx