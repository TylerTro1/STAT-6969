''' This file contains the functions which return the best pre-processing / model / hyperparameter combinations for a given dataset.
'''

from model import LogisticRegression, Model, SupportVectorMachine, MODEL_OPTIONS
from cross_validation import cross_validation
from utils import accuracy

# Some imports yet to be implemented
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




def preprocessing_pipeline(preprocessors: List[str], preprocessor_params: Dict[str, Dict] = None) -> Pipeline:
    """
    Build a preprocessing pipeline based on the specified preprocessors.
    
    Args:
        preprocessors: List of preprocessor names to include in the pipeline
        preprocessor_params: Dictionary of parameters for each preprocessor
        
    Returns:
        sklearn.Pipeline with the specified preprocessing steps
    """
    if preprocessor_params is None:
        preprocessor_params = {}
        
    steps = []
    
    available_preprocessors = {
        'standard_scaler': ('scaler', StandardScaler()),
        'robust_scaler': ('robust_scaler', RobustScaler()),
        'minmax_scaler': ('minmax_scaler', MinMaxScaler()),
        'variance_threshold': ('variance_threshold', VarianceThreshold(threshold=0.01)),
        'select_kbest': ('select_kbest', SelectKBest(f_classif)),
        'pca': ('pca', PCA()),
        'polynomial_features': ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
        'smote': ('smote', SMOTE(random_state=42))
    }
    
    for preprocessor in preprocessors:
        if preprocessor in available_preprocessors:
            name, processor = available_preprocessors[preprocessor]
            steps.append((name, processor))
    
    if 'smote' in [p for p in preprocessors]:
        return ImbPipeline(steps)
    return Pipeline(steps)







def get_model_class(model_name: str) -> BaseEstimator:
    """
    Get the model class based on the model name.
    
    Args:
        model_name: Name of the model to retrieve
        
    Returns:
        Model class
    """

    model_map = {
        'svm': SupportVectorMachine,
        'logistic_regression': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'xgboost': XGBClassifier,
        'mlp': MLPClassifier
    }
    return model_map.get(model_name.lower())







def get_model_param_grid(model_name: str) -> Dict:
    """
    Get default parameter grid for the specified model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary of parameter grid for GridSearchCV
    """
    param_grids = {
        'svm': {
            'classifier__lr0': [0.1, 0.01, 0.001],
            'classifier__C': [0.1, 1, 10]
        },
        'logistic_regression': {
            'classifier__lr0': [0.1, 0.01, 0.001],
            'classifier__sigma2': [0.1, 1, 10]
        },
        'random_forest': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5]
        },
        'xgboost': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.05, 0.1]
        },
        'mlp': {
            'classifier__hidden_layer_sizes': [(50,), (100,)],
            'classifier__activation': ['relu', 'tanh'],
            'classifier__solver': ['adam', 'sgd']
        }
    }
    return param_grids.get(model_name.lower(), {})




def optimize_model_pipeline(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    model_names: List[str],
    preprocessor_combinations: List[List[str]],
    cv: int = 5,
    scoring: str = 'f1',
    n_jobs: int = -1,
    verbose: int = 1
) -> Tuple[Pipeline, Dict, Dict]:
    """
    Optimize model pipelines with different preprocessing combinations and hyperparameters.
    
    Args:
        X: Feature data (DataFrame or array)
        y: Target labels (Series or array)
        model_names: List of model names to evaluate
        preprocessor_combinations: List of lists of preprocessor combinations to try
        cv: Number of cross-validation folds
        scoring: Scoring metric to optimize
        n_jobs: Number of jobs to run in parallel
        verbose: Verbosity level
        
    Returns:
        Tuple containing:
        - Best pipeline
        - Best parameters
        - All results (models, parameters, scores)
    """

    results = []
    best_score = -np.inf
    best_pipeline = None
    best_params = None
    
    scorers = {
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score),
        'f1_weighted': make_scorer(f1_score, average='weighted')
    }
    
    for model_name in model_names:
        model_class = get_model_class(model_name)
        if model_class is None:
            raise Exception("Error--Model is None")
            
        for preprocessors in preprocessor_combinations:
            try:
                pipeline = preprocessing_pipeline(preprocessors)
                pipeline.steps.append(('classifier', model_class()))
                
                param_grid = get_model_param_grid(model_name)
                
                if 'variance_threshold' in preprocessors:
                    param_grid['variance_threshold__threshold'] = [0.01, 0.05, 0.1]
                if 'select_kbest' in preprocessors:
                    param_grid['select_kbest__k'] = [50, 100, 'all']
                if 'pca' in preprocessors:
                    param_grid['pca__n_components'] = [0.95, 0.99, 50, 100]
                
                grid_search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid,
                    cv=cv,
                    scoring=scorers,
                    refit=scoring,
                    n_jobs=n_jobs,
                    verbose=verbose
                )
                
                grid_search.fit(X, y)
                
                result = {
                    'model': model_name,
                    'preprocessors': preprocessors,
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'cv_results': grid_search.cv_results_,
                    'grid_search': grid_search
                }
                results.append(result)
                
                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_pipeline = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    
                print(f"Completed: Model={model_name}, Preprocessors={preprocessors}, Best {scoring}={grid_search.best_score_:.4f}")
                
            except Exception as e:
                print(f"Error with Model={model_name}, Preprocessors={preprocessors}: {e}")
                continue
                
    return best_pipeline, best_params, results
