''' main -- to be run
'''
# Some imports yet implemented
import pandas as pd

from final_project_functions import optimize_model_pipeline, preprocessing_pipeline, get_model_class, get_model_param_grid

from model import LogisticRegression, Model, SupportVectorMachine, MODEL_OPTIONS
from cross_validation import cross_validation
from utils import accuracy

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score, confusion_matrix, classification_report, make_scorer, accuracy_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier, MLPRegressor
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression as SkLogReg
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline





if __name__ == "__main__":
    ### LOAD DATA ###
    train_data = pd.read_csv(r"D:\STAT 6969\project\project_data\data\train.csv") # <--- CHANGE THIS to fit your directory
    test_data = pd.read_csv(r"D:\STAT 6969\project\project_data\data\test.csv") # <--- CHANGE THIS to fit your directory

    train_labels = train_data['label']
    test_labels = test_data['label']

    train_data = train_data.drop(columns = 'label')
    test_data = test_data.drop(columns = 'label')



    ### REMOVE COLUMNS OF ZEROES ###
    cols_to_drop = train_data.columns[(train_data == 0).all()]

    train_data = train_data.drop(cols_to_drop, axis = 1)
    test_data = test_data.drop(cols_to_drop, axis = 1)


    # Define models and preprocessor combinations to try
    models_to_try = ['svm', 'logistic_regression', 'random_forest', 'xgboost']
    
    preprocessor_combinations = [
        ['standard_scaler', 'minmax scaler', 'pca', 'smote', 'variance_threshold', 'select_kbest'],
        ['standard_scaler', 'minmax scaler', 'pca', 'smote', 'variance_threshold', 'select_kbest'],
        ['standard_scaler', 'minmax scaler', 'pca', 'smote', 'variance_threshold', 'select_kbest'],
        ['standard_scaler', 'minmax scaler', 'pca', 'smote', 'variance_threshold', 'select_kbest'],
        ['standard_scaler', 'minmax scaler', 'pca', 'smote', 'variance_threshold', 'select_kbest']
    ]
    
    X = train_data.values if isinstance(train_data, pd.DataFrame) else train_data
    y = train_labels.values if isinstance(train_labels, pd.Series) else train_labels
    
    best_pipeline, best_params, all_results = optimize_model_pipeline(
        X=X,
        y=y,
        model_names=models_to_try,
        preprocessor_combinations=preprocessor_combinations,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    print("\n=== BEST PIPELINE ===")
    print(best_pipeline)
    print("\n=== BEST PARAMETERS ===")
    print(best_params)
    
    if test_data is not None and test_labels is not None:
        X_test = test_data.values if isinstance(test_data, pd.DataFrame) else test_data
        y_test = test_labels.values if isinstance(test_labels, pd.Series) else test_labels
        
        y_pred = best_pipeline.predict(X_test)
        print("\n=== TEST SET PERFORMANCE ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))