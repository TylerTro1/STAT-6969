''' This file contains utility functions.
'''

from typing import Tuple

import numpy as np


def shuffle_data(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Helper function to shuffle two np.ndarrays s.t. if x[i] <- x[j] after shuffling,
    y[i] <- y[j] after shuffling for all i, j.

    Args:
        x (np.ndarray): the first array
        y (np.ndarray): the second array

    Returns
        (np.ndarray, np.ndarray): tuple of shuffled x and y
    '''

    assert len(x) == len(y), f'{len(x)=} and {len(y)=} must have the same length in dimension 0'
    p = np.random.permutation(len(x))
    return x[p], y[p]


def clip(x: np.ndarray, max_abs_value: float = 10000) -> np.ndarray:
    '''
    Helper function for clipping very large (or very small values)

    Args:
        x (np.ndarray): the value to be clipped. Can be an np.ndarray or a single float.
        max_abs_value (float): the maximum value that |x| will have after clipping s.t. -max_abs_value <= x <= max_abs_value

    Returns:
        np.ndarray: an np.ndarray containing the clipped values. Will be a float if x is a float.
    '''

    return np.minimum(np.maximum(x, -abs(max_abs_value)), abs(max_abs_value))



def accuracy(labels: list, predictions: list) -> float:
    '''
    Calculate the accuracy between ground-truth labels and candidate predictions.
    Should be a float between 0 and 1.

    Args:
        labels (list): the ground-truth labels from the data
        predictions (list): the predicted labels from the model

    Returns:
        float: the accuracy of the predictions, when compared to the ground-truth labels
    '''

    assert len(labels) == len(predictions), (
        f'{len(labels)=} and {len(predictions)=} must be the same length.' + 
        '\n\n  Have you implemented model.predict()?\n'
    )
    
    correct = 0
    for label, prediction in zip(labels, predictions):
        if label == prediction:
            correct += 1
    accuracy = correct / len(labels)

    return accuracy
