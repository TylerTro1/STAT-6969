''' This file contains the functions for performing cross-validation.
'''

import argparse
import itertools
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from utils import accuracy
from model import LogisticRegression, Model, SupportVectorMachine, MODEL_OPTIONS


def init_model(model_name: str, lr0: float, reg_tradeoff: float, num_features: int) -> Model:
    '''
    Initialize the appropriate model with corresponding hyperparameters.

    Args:
        model_name (str): which model to use. Should be either "svm" or "logistic_regression"
        lr0 (float): the initial learning rate (gamma_0)
        reg_tradeoff (float): the regularization/loss tradeoff hyperparameter for SVM and Logistic Regression
        num_features (int): the number of features (i.e. dimensions) the model will have

    Returns:
        Model: a Model object initialized with the corresponding hyperparameters
    '''

    if model_name == 'svm':
        model = SupportVectorMachine(num_features=num_features, lr0=lr0, C=reg_tradeoff)
    elif model_name == 'logistic_regression':
        model = LogisticRegression(num_features=num_features, lr0=lr0, sigma2=reg_tradeoff)
    return model


def cross_validation(
        cv_folds: List[pd.DataFrame], 
        model_name: str, 
        lr0_values: list, 
        reg_tradeoff_values: list,
        epochs: int = 5) -> Tuple[dict, float]:
    '''
    Run cross-validation to determine the best hyperparameters.

    Args:
        cv_folds (list): a list of pandas DataFrames, corresponding to folds of the data. 
            The last column of each DataFrame, called "label", corresponds to y, 
            while the remaining columns are the features x.
        model_name (str): which model to use. Should be either "svm" or "logistic_regression"
        lr0_values (list): a list of initial learning rate values to try. 
            Equivalent to "C" hyperparam for SVM, or "sigma2" hyperparam for Logistic Regression.
        reg_tradeoff_values (list): a list of regularization tradeoff values to try.
        epochs (int): how many epochs to train each model for. Defaults to 5

    Returns:
        dict: a dictionary with the best hyperparameters discovered during cross-validation
        float: the average cross-validation accuracy corresponding to the best hyperparameters

    Hints:
        - We've provided a helper function `init_model()` above to initialize your model. 
        - The python `itertools.product()` function returns the Cartesian product of multiple lists.
        - You can convert a pandas DataFrame to a numpy ndarray with `df.to_numpy()`
    '''

    best_hyperparams = {'lr0': None, 'reg_tradeoff': None}
    best_avg_accuracy = 0

    for lr0, reg_tradeoff in itertools.product(lr0_values, reg_tradeoff_values):
        fold_accuracies = []

        for i in range(len(cv_folds)): 
            validation_fold = cv_folds[i]
            training_folds = [fold for j, fold in enumerate(cv_folds) if j != i]

            train_data = pd.concat(training_folds)

            y_train = train_data['label'].to_numpy()
            x_train = train_data.drop('label', axis = 1).to_numpy()

            y_val = validation_fold['label'].to_numpy()
            x_val = validation_fold.drop('label', axis = 1).to_numpy()

            model = init_model(model_name, lr0, reg_tradeoff, train_data.shape[1] - 1)
            model.train(x_train, y_train, epochs)

            predictions = model.predict(x_val)
            fold_accuracy = accuracy(y_val, predictions)
            fold_accuracies.append(fold_accuracy)

            avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)

            if avg_accuracy > best_avg_accuracy: 
                best_avg_accuracy = avg_accuracy
                best_hyperparams['lr0'] = lr0
                best_hyperparams['reg_tradeoff'] = reg_tradeoff

    return best_hyperparams, best_avg_accuracy



def cross_validation_with_plotting(
        cv_folds: List[pd.DataFrame], 
        model_name: str, 
        lr0_values: list, 
        reg_tradeoff_values: list,
        epochs: int = 5) -> Tuple[dict, float, plt.Figure]:
    '''
    Run cross-validation and return results for plotting.
    
    Returns:
        dict: best hyperparameters
        float: best accuracy
        plt.Figure: cross-validation plot figure
    '''
    results = {lr0: {'reg_values': [], 'accuracies': []} for lr0 in lr0_values}
    
    best_hyperparams = {'lr0': None, 'reg_tradeoff': None}
    best_avg_accuracy = 0

    for lr0, reg_tradeoff in itertools.product(lr0_values, reg_tradeoff_values):
        fold_accuracies = []

        for i in range(len(cv_folds)): 
            validation_fold = cv_folds[i]
            training_folds = [fold for j, fold in enumerate(cv_folds) if j != i]

            train_data = pd.concat(training_folds)
            y_train = train_data['label'].to_numpy()
            x_train = train_data.drop('label', axis=1).to_numpy()
            y_val = validation_fold['label'].to_numpy()
            x_val = validation_fold.drop('label', axis=1).to_numpy()

            model = init_model(model_name, lr0, reg_tradeoff, train_data.shape[1] - 1)
            model.train(x_train, y_train, epochs)

            predictions = model.predict(x_val)
            fold_accuracy = accuracy(y_val, predictions)
            fold_accuracies.append(fold_accuracy)

        avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
        results[lr0]['reg_values'].append(reg_tradeoff)
        results[lr0]['accuracies'].append(avg_accuracy)

        if avg_accuracy > best_avg_accuracy: 
            best_avg_accuracy = avg_accuracy
            best_hyperparams['lr0'] = lr0
            best_hyperparams['reg_tradeoff'] = reg_tradeoff

    # Create the plot
    plt.figure(figsize=(10, 6))
    for lr0 in lr0_values:
        sorted_data = sorted(zip(results[lr0]['reg_values'], results[lr0]['accuracies']), key=lambda x: x[0])
        reg_values = [x[0] for x in sorted_data]
        accuracies = [x[1] for x in sorted_data]
        
        plt.plot(reg_values, accuracies, 'o-', label=f'lr0={lr0}')
    
    plt.xscale('log')
    plt.xlabel('Regularization Tradeoff Parameter (log scale)')
    plt.ylabel('Average Cross-Validation Accuracy')
    plt.title(f'Cross-Validation Results for {model_name}')
    plt.legend()
    plt.grid(True)
    
    return best_hyperparams, best_avg_accuracy, plt.gcf()

