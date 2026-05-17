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
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        prev = nx
        for l in range(1, self.L + 1):
            self.weights['W' + str(l)] = (
                np.random.randn(layers[l - 1], prev) * np.sqrt(2 / prev)
            )
            self.weights['b' + str(l)] = np.zeros((layers[l - 1], 1))
            prev = layers[l - 1]
            