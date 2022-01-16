import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)

class LogisticRegression:
    
    def __init__(self, X, is_linear = True):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.theta = np.zeros(X.values.shape[-1]+1)                                  # Random numpy float array for theta weights
        self.alpha = 0.01                                                            # Learning rate
        self.epochs = 300                                                            # Number of iterations
        self.is_linear = is_linear
        self.origo = [0, 0]

    
    def fit(self, X, Y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        trans_X = np.concatenate((np.ones((X.values.shape[0], 1)), X.values), axis=1)
        # Assuming the data is linear
        if self.is_linear:
            #θ := θ + α( y^i − h_θ(x^i) ) * x^i
            for epoch in range(self.epochs): # Batch Gradient Descent
                y_pred = self.predict(X)
                self.theta += self.alpha * np.dot(trans_X.T, (Y - y_pred))
        
        #Data is circular (non-linear)
        else:
            self.origo = np.mean(X.values, axis=0)
            X_r = get_radius(X.values, self.origo)
            self.theta = np.zeros(2) 
            for epoch in range(self.epochs): # Batch Gradient Descent
                y_pred = self.predict(X)
                self.theta += self.alpha * np.dot(X_r.T, (Y - y_pred))
                
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        if isinstance(X, pd.DataFrame) and self.is_linear:
            X = np.concatenate((np.ones((X.values.shape[0], 1)), X.values), axis=1)
            
        elif not isinstance(X, pd.DataFrame) and self.is_linear:
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
            
        if not self.is_linear:
            X = get_radius(X.values, self.origo)
            
        z = np.dot(X, self.theta)
        return np.array(sigmoid(z))
        

        
# --- Some utility functions 

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )

def sigmoid(z):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-z))

def get_radius(X, origo):
    X_r = np.sqrt(np.square(X[:, 0] - origo[0]) + np.square(X[:, 1] - origo[1])).reshape(X.shape[0], -1)
    X_r = np.concatenate((np.ones((X_r.shape[0], 1)), X_r), axis=1) #Add n * 1 for const
    return X_r

        