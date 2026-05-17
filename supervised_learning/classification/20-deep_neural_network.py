#!/usr/bin/env python3
"""Module that defines a deep neural network for binary classification."""

import numpy as np


class DeepNeuralNetwork:
    """A deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """Initialize the deep neural network.

        Args:
            nx (int): number of input features.
            layers (list): number of nodes in each layer.

        Raises:
            TypeError: if nx is not an integer.
            ValueError: if nx is less than 1.
            TypeError: if layers is not a list of positive integers.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        prev = nx
        for l in range(1, self.__L + 1):
            self.__weights['W' + str(l)] = (
                np.random.randn(layers[l - 1], prev) * np.sqrt(2 / prev)
            )
            self.__weights['b' + str(l)] = np.zeros((layers[l - 1], 1))
            prev = layers[l - 1]

    @property
    def L(self):
        """Getter for L."""
        return self.__L

    @property
    def cache(self):
        """Getter for cache."""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights."""
        return self.__weights

    def forward_prop(self, X):
        """Calculate the forward propagation of the neural network.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).

        Returns:
            tuple: output of the network and the cache.
        """
        self.__cache['A0'] = X
        A = X
        for l in range(1, self.__L + 1):
            W = self.__weights['W' + str(l)]
            b = self.__weights['b' + str(l)]
            Z = np.matmul(W, A) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache['A' + str(l)] = A
        return A, self.__cache

    def cost(self, Y, A):
        """Calculate the cost using logistic regression.

        Args:
            Y (numpy.ndarray): correct labels of shape (1, m).
            A (numpy.ndarray): activated output of shape (1, m).

        Returns:
            float: the logistic regression cost.
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluate the neural network's predictions.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).
            Y (numpy.ndarray): correct labels of shape (1, m).

        Returns:
            tuple: predicted labels of shape (1, m) and the cost.
        """
        A, _ = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculate one pass of gradient descent on the neural network.

        Args:
            Y (numpy.ndarray): correct labels of shape (1, m).
            cache (dict): intermediary values of the network.
            alpha (float): learning rate.
        """
        m = Y.shape[1]
        dZ = cache['A' + str(self.__L)] - Y
        for l in range(self.__L, 0, -1):
            A_prev = cache['A' + str(l - 1)]
            W = self.__weights['W' + str(l)]
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            if l > 1:
                A_prev_val = cache['A' + str(l - 1)]
                dZ = np.matmul(W.T, dZ) * (A_prev_val * (1 - A_prev_val))
            self.__weights['W' + str(l)] -= alpha * dW
            self.__weights['b' + str(l)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Train the deep neural network.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).
            Y (numpy.ndarray): correct labels of shape (1, m).
            iterations (int): number of training iterations.
            alpha (float): learning rate.

        Raises:
            TypeError: if iterations is not an integer.
            ValueError: if iterations is not positive.
            TypeError: if alpha is not a float.
            ValueError: if alpha is not positive.

        Returns:
            tuple: evaluation of training data after training.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for _ in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
        return self.evaluate(X, Y)
    