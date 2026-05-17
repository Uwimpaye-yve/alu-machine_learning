#!/usr/bin/env python3
"""Module for one-hot encoding of numeric label vectors."""

import numpy as np


def one_hot_encode(Y, classes):
    """Convert a numeric label vector into a one-hot matrix.

    Args:
        Y (numpy.ndarray): numeric class labels of shape (m,).
        classes (int): maximum number of classes.

    Returns:
        numpy.ndarray: one-hot encoding of shape (classes, m), or None.
    """
    if not isinstance(Y, np.ndarray) or len(Y) == 0:
        return None
    if not isinstance(classes, int) or classes < 2:
        return None
    if classes <= np.max(Y):
        return None
    try:
        m = Y.shape[0]
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
        return one_hot
    except Exception:
        return None
    