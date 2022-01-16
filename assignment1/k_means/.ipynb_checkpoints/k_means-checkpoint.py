import numpy as np 
import pandas as pd 
import random
from copy import deepcopy
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self, k, transform = False):
        self.k = k
        self.epochs = 10
        self.centroids = np.zeros(shape=(self.k, 2))
        self.transform= transform
        
    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        #Transform to more circulare clusters
        if self.transform:
            X = transform(X)
            
        min_distortion = float('inf')
        best_clusters = np.zeros(len(X.index))
        for epoch in range(self.epochs): # Find the best local minimum
            
            self.k_means_plus(X)                
            epsilon = 1.00E-6
            error = float('inf')
            while error > epsilon: #Find new mean until convergence
                
                cluster_assign = np.zeros(len(X.index))
                cluster_map = [[] for _ in range(len(self.centroids))]
                for row, sample in enumerate(X.values): #Assign all samples to nearest centroid
                    min_dist = [None, float('inf')]
                    for i, centroid in enumerate(self.centroids):
                        new_dist = euclidean_distance(sample, centroid)
                        if new_dist < min_dist[1]: 
                            min_dist = [i, new_dist]
                    cluster_assign[row] = min_dist[0]
                    cluster_map[min_dist[0]].append(sample)

                new_centroids = np.zeros(shape=(self.k, 2))
                for i, centroid in enumerate(cluster_map): #Calculate new_centroid := mean_value
                    new_centroids[i] = np.mean(centroid, axis=0)
                        
                error = 0
                for i, centroid in enumerate(self.centroids): # Check convergence
                    error += euclidean_distance(centroid, new_centroids[i])

                self.centroids = new_centroids
                
            new_distortion = euclidean_distortion(X, cluster_assign.astype(int)) # Update i better centroids are found
            if new_distortion < min_distortion:
                min_distortion = new_distortion
                best_clusters = cluster_assign.astype(int)
                best_centroids = deepcopy(self.centroids)
                
        self.centroids = best_centroids
                
                
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        epsilon = 1.00E-6
        error = float('inf')
        while error > epsilon: #Find new mean until convergence
            cluster_assign = np.zeros(len(X.index))
            cluster_map = [[] for _ in range(len(self.centroids))]

            for row, sample in enumerate(X.values): #Assign all samples to nearest centroid
                min_dist = [None, float('inf')]
                for i, centroid in enumerate(self.centroids):
                    new_dist = euclidean_distance(sample, centroid)
                    if new_dist < min_dist[1]: 
                        min_dist = [i, new_dist]
                cluster_assign[row] = min_dist[0]
                cluster_map[min_dist[0]].append(sample)

            new_centroids = np.zeros(shape=(self.k, 2))
            for i, centroid in enumerate(cluster_map): #Calculate new_centroid := mean_value
                new_centroids[i] = np.mean(centroid, axis=0)
                        
            error = 0
            for i, centroid in enumerate(self.centroids):
                error += euclidean_distance(centroid, new_centroids[i])
        return cluster_assign.astype(int);
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.centroids

    def k_means_plus(self, X):
        self.centroids[0] = X.sample(replace = True).values[0] #First centroid
        for chosen_centroid in range(1, self.k):
            min_dist_array = np.zeros(len(X.values))
            for row, sample in enumerate(X.values): #Assign all samples to nearest centroid
                min_dist = np.array([None, float('inf')])
                for i in range(chosen_centroid):
                    centroid = self.centroids[i]
                    new_dist = euclidean_distance(sample, centroid)
                    if new_dist < min_dist[1]: 
                            min_dist = [i, new_dist]
                min_dist_array[row] = min_dist[1]
            weights = np.square(min_dist_array)/np.sum(np.square(min_dist_array)) #Probability based on how far away they from the closest cluster
            new_centroid = np.random.choice(len(X.values), p = weights)
            self.centroids[chosen_centroid] = X.values[new_centroid]
                

    
    
# --- Some utility functions 

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    
    
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
        
    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))

def transform(X):
    delta_x = X.values.max(0)[0] - X.values.min(0)[0]
    delta_y = X.values.max(0)[1] - X.values.min(0)[1]
    scale_y = delta_x/delta_y # y * scale = x => scale = x / y
    X[X.columns[1]] = scale_y * X[X.columns[1]]
    return X



