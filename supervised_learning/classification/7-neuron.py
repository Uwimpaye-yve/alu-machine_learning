#!/usr/bin/env python3
"""Module that defines a single neuron performing binary classification."""

import matplotlib.pyplot as plt
import numpy as np


class Neuron:
    """A single neuron performing binary classification."""

    def __init__(self, nx):
        """Initialize the neuron.

        Args:
            nx (int): number of input features to the neuron.

        Raises:
            TypeError: if nx is not an integer.
            ValueError: if nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights vector."""
        return self.__W

    @property
    def b(self):
        """Getter for bias."""
        return self.__b

    @property
    def A(self):
        """Getter for activated output."""
        return self.__A

    def forward_prop(self, X):
        """Calculate the forward propagation of the neuron.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).

        Returns:
            numpy.ndarray: the activated output __A.
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

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
        """Evaluate the neuron's predictions.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).
            Y (numpy.ndarray): correct labels of shape (1, m).

        Returns:
            tuple: predicted labels of shape (1, m) and the cost.
        """
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, self.cost(Y, A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculate one pass of gradient descent on the neuron.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).
            Y (numpy.ndarray): correct labels of shape (1, m).
            A (numpy.ndarray): activated output of shape (1, m).
            alpha (float): learning rate.
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = np.matmul(dZ, X.T) / m
        db = np.sum(dZ) / m
        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Train the neuron.

        Args:
            X (numpy.ndarray): input data of shape (nx, m).
            Y (numpy.ndarray): correct labels of shape (1, m).
            iterations (int): number of training iterations.
            alpha (float): learning rate.
            verbose (bool): whether to print cost during training.
            graph (bool): whether to plot cost after training.
            step (int): interval for verbose/graph output.

        Raises:
            TypeError: if iterations is not an integer.
            ValueError: if iterations is not positive.
            TypeError: if alpha is not a float.
            ValueError: if alpha is not positive.
            TypeError: if step is not an integer (when verbose or graph).
            ValueError: if step is not positive or > iterations.

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
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iters = []

        for i in range(iterations + 1):
            if i < iterations:
                A = self.forward_prop(X)
            if verbose and (i % step == 0 or i == iterations):
                c = self.cost(Y, self.__A)
                print("Cost after {} iterations: {}".format(i, c))
            if graph and (i % step == 0 or i == iterations):
                costs.append(self.cost(Y, self.__A))
                iters.append(i)
            if i < iterations:
                self.gradient_descent(X, Y, A, alpha)

        if graph:
            plt.plot(iters, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
    