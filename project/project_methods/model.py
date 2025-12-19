''' This file defines the model classes that will be used. 
'''

from typing import Protocol

import numpy as np
import matplotlib.pyplot as plt

from utils import clip, shuffle_data, accuracy



# set the numpy random seed so our randomness is reproducible
np.random.seed(1)

MODEL_OPTIONS = ['majority_baseline', 'svm', 'logistic_regression']











class Model(Protocol):
    def __init__(**hyperparam_kwargs):
        ...

    def get_hyperparams(self) -> dict:
        ...

    def loss(self, ) -> float:
        ...
        
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        ...

    def predict(self, x: np.ndarray) -> list:
        ...
















class MajorityBaseline(Model):
    def __init__(self):
        
        self.majority_label = None


    def get_hyperparams(self) -> dict:
        return {}
    

    def loss(self, x_i: np.ndarray, y_i: int) -> float:
        return None
    

    def train(self, x: np.ndarray, y: np.ndarray):
        '''
        Train a baseline model that returns the most common label in the dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
        '''
    
        unique_labels, counts = np.unique(y, return_counts = True)
        self.majority_label = unique_labels[np.argmax(counts)]




    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        return [self.majority_label] * len(x)
    





























class SupportVectorMachine(Model):
    def __init__(self, num_features: int, lr0: float, C: float):
        '''
        Initialize a new SupportVectorMachine model

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            lr0 (float): the initial learning rate (gamma_0)
            C (float): the regularization/loss tradeoff hyperparameter
            w (np.ndarray): the weights vector
        '''

        self.lr0 = lr0
        self.C = C
        self.w = np.random.normal(-0.01, 0.01, num_features)
        self.b = 0.0
        self.epoch_losses = []
        self.name = 'SVM'

    
    def get_hyperparams(self) -> dict:
        return {'lr0': self.lr0, 'C': self.C}
    

    def loss(self, x_i: np.ndarray, y_i: int) -> float:
        '''
        Calculate the SVM loss on a single example.

        Args:
            x_i (np.ndarray): a 1-D np.ndarray (num_features) with the features for a single example
            y_i (int): the label for the example, either 0 or 1.

        Returns:
            float: the loss for the example using the current weights
        '''

        y_i = 1 if y_i == 1 else -1
        mu = y_i * (np.dot(self.w, x_i) + self.b)
        hinge_loss = max(0.0, 1 - mu)
        return hinge_loss
    
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for
        '''

        num_examples = x.shape[0]
        self.epoch_losses = []
        

        for epoch in np.arange(epochs):
            x_shuffled, y_shuffled = shuffle_data(x, y)
            total_loss = 0

            for i in range(num_examples):
                x_i = x_shuffled[i]
                y_i = y_shuffled[i]
                y_i = 2 * y_i - 1

                gamma_t = self.lr0 / (1 + epoch)
                margin = y_i * (np.dot(self.w, x_i) + self.b)

                if margin < 1: 
                    self.w = (1 - gamma_t) * self.w + gamma_t * self.C * y_i * x_i
                    self.b += gamma_t * self.C * y_i
                else: 
                    self.w = (1 - gamma_t) * self.w
                
                total_loss += self.loss(x_shuffled[i], y_shuffled[i])

            avg_loss = total_loss / num_examples
            self.epoch_losses.append(avg_loss)



    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        scores = np.dot(x, self.w) + self.b
        return (scores >= 0).astype(int).tolist()

























class LogisticRegression(Model):
    def __init__(self, num_features: int, lr0: float, sigma2: float):
        '''
        Initialize a new LogisticRegression model

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            lr0 (float): the initial learning rate (gamma_0)
            sigma2 (float): the regularization/loss tradeoff hyperparameter
        '''

        self.lr0 = lr0
        self.sigma2 = sigma2
        self.w = np.zeros(num_features)
        self.b = 0.0
        self.epoch_losses = []
        self.name = 'Logistic Regression'

    
    def get_hyperparams(self) -> dict:
        return {'lr0': self.lr0, 'sigma2': self.sigma2}
    

    def loss(self, x_i: np.ndarray, y_i: int) -> float:
        '''
        Calculate the SVM loss on a single example.

        Args:
            x_i (np.ndarray): a 1-D np.ndarray (num_features) with the features for a single example
            y_i (int): the label for the example, either 0 or 1.

        Returns:
            float: the loss for the example using the current weights
        '''

        z = clip(np.dot(self.w, x_i) + self.b, 100)  # Already clipped to prevent overflow
        sig = sigmoid(z)
        
        epsilon = 1e-5
        sig = np.clip(sig, epsilon, 1-epsilon)
        
        log_loss = -y_i * np.log(sig) - (1 - y_i) * np.log(1 - sig)
        
        reg_loss = (1 / (2 * self.sigma2)) * np.sum(self.w**2)
        
        return log_loss + reg_loss



        #Test
        # log_loss = np.log(1 + np.exp(-y_i * np.dot(self.w, x_i)))
        # reg_loss = np.dot(self.w, self.w) / self.sigma2**2


        # return log_loss + reg_loss
    
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for
        '''

        num_examples = x.shape[0]
        self.epoch_losses = []

        for epoch in np.arange(epochs):
            x_shuffled, y_shuffled = shuffle_data(x, y)
            total_loss = 0

            for i in range(num_examples): 
                x_i = x_shuffled[i]
                y_i = y_shuffled[i]

                gamma_t = self.lr0 / (1 + epoch)

                z = clip(np.dot(self.w, x_i) + self.b, 100)
                sig = sigmoid(z)

                diff = sig - y_i
                grad_w = diff * x_i + (self.w / self.sigma2)
                grad_b = diff

                self.w -= gamma_t * grad_w
                self.b -= gamma_t * grad_b 


                #Test
                # z = clip(-y_i * np.dot(self.w, x_i) + self.b, 100)
                # sig = sigmoid(z)

                # diff = -y_i*x_i*sig + 2*self.w / self.sigma2**2
                # self.w += gamma_t * diff

                total_loss += self.loss(x_i, y_i)

            avg_loss = total_loss / num_examples
            self.epoch_losses.append(avg_loss)


    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''
        x = np.asarray(x, dtype=np.float64)

        predictions = []

        for x_i in x: 
            p = sigmoid(np.dot(self.w, x_i) + self.b)
            predictions.append(1 if p >= 0.5 else 0)
        return predictions















def plot_training_loss(model: Model, dataset_name: str) -> plt.Figure:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(model.epoch_losses) + 1), model.epoch_losses, 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('Average Training Loss')
    plt.title(f'Training Loss per Epoch with {model.name}')
    plt.grid(True)
    return plt.gcf()








def sigmoid(z: float) -> float:
    '''
    The sigmoid function.

    Args:
        z (float): the argument to the sigmoid function.

    Returns:
        float: the sigmoid applied to z.          
    '''
    
    z = clip(z, 100)
    return 1 / (1 + np.exp(-z))