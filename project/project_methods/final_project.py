''' This file constitutes the brute-force pre-processing done to evaluate a dataset of interest with discrete, numerical features and binary (-1,1) labels
'''

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






### LOAD DATA ###
train_data = pd.read_csv(r"D:\STAT 6969\project\project_data\data\train.csv") # <--- CHANGE THIS to fit your stored location for this file
test_data = pd.read_csv(r"D:\STAT 6969\project\project_data\data\test.csv") # <--- CHANGE THIS to fit your stored location for this file

train_labels = train_data['label']  #['auto_parts']
test_labels = test_data['label']

fprint(f'Train Data Columns: {train_labels.value_counts}')

train_data = train_data.drop(columns = 'label')
test_data = test_data.drop(columns = 'label')



### REMOVE COLUMNS OF ZEROES ###
cols_to_drop = train_data.columns[(train_data == 0).all()]

train_data = train_data.drop(cols_to_drop, axis = 1)
test_data = test_data.drop(cols_to_drop, axis = 1)




print(train_data.shape)
print(test_data.shape)

# print(train_data.head())





######### PRE-PROCESSING APPROACHES ########

### VARIANCE THRESHOLD APPROACH ###
selector = VarianceThreshold(threshold = 0.01)

train_data_reduced = selector.fit_transform(train_data)
train_data_variance_threshold = pd.DataFrame(train_data_reduced, columns = train_data.columns[selector.get_support()])

# test_data_reduced = selector.fit_transform(test_data)
test_data_variance_threshold = pd.DataFrame(test_data, columns = test_data.columns[selector.get_support()])

# print(train_data)




### THRESHOLD APPROACH ###
threshold = 0.95
train_ratio = (train_data != 0).sum() / len(train_data)
test_ratio = (test_data != 0).sum() / len(test_data)

train_data_threshold = train_data.loc[:, train_ratio > (1 - threshold)]

test_data_threshold = test_data.loc[:, test_ratio > (1 - threshold)]

# print(train_data)





### STANDARD SCALER APPROACH ###
standard_scaler = StandardScaler()

standard_scaler.fit(train_data)

train_data_scaled = standard_scaler.fit_transform(train_data)
train_data_standard_scaler = pd.DataFrame(train_data_scaled, columns = train_data.columns)

test_data_scaled = standard_scaler.transform(test_data)
test_data_standard_scaler = pd.DataFrame(test_data_scaled, columns = test_data.columns)

# print(train_data)
# print(train_data.shape)





### MIN-MAX SCALER APPROACH ###
minmax_scaler = MinMaxScaler()

train_data_minmax = pd.DataFrame(minmax_scaler.fit_transform(train_data), columns=train_data.columns)
test_data_minmax = pd.DataFrame(minmax_scaler.transform(test_data), columns=test_data.columns)





### PCA APPROACH ###
pca = PCA(n_components=0.95)  # keep 95%
train_data_pca = pd.DataFrame(pca.fit_transform(train_data))
test_data_pca = pd.DataFrame(pca.transform(test_data))

# print(train_data.shape)




### POLYNOMIAL FEATURES ###
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

train_data_poly = pd.DataFrame(poly.fit_transform(train_data))
test_data_poly = pd.DataFrame(poly.transform(test_data))


print(train_data.shape)




#SelectKBest
k_best_selector = SelectKBest(score_func = f_classif, k=50)
train_selected = selector.fit_transform(train_data_standard_scaler, train_labels)
test_selected = selector.transform(test_data_standard_scaler)

svm = SupportVectorMachine(num_features=train_selected.shape[1], lr0 = 0.01, C=1.0)
svm.train(train_selected, train_labels.to_numpy(), epochs = 10)

predictions = svm.predict(test_selected)



print(f'K-BEST ACC EVAL: {accuracy(test_labels, predictions)} \n')










# Preprocessing
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=50)),
])

X_train_transformed = pipeline.fit_transform(train_data)
test_transformed = pipeline.transform(test_data)


#Ensemble
base_models = [
    ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="logloss")),
    ("rf", RandomForestClassifier(n_estimators=100))
]

ensemble = StackingClassifier(
    estimators=base_models,
    final_estimator=SkLogReg()
)

ensemble.fit(X_train_transformed, train_labels)
y_pred_ensemble = ensemble.predict(test_transformed)


print(f'ACC FOR ENSEMBLE (XGBoost & RandomForest): {accuracy(test_labels, y_pred_ensemble)} \n')






























### PERFORM TESTING TO SEE WHICH WORKS BEST (Using best known hyper-parameters)###

#Basic
SVM_basic = SupportVectorMachine(train_data.shape[1], 0.001, 10)
SVM_basic.train(train_data.to_numpy(), train_labels.to_numpy(), 15)
test_data_basic_predicted_labels_SVM = SVM_basic.predict(test_data.to_numpy())

print('------BASIC: ------ \n')

print(f'BASIC TEST ACC SVM: {accuracy(test_labels, test_data_basic_predicted_labels_SVM)} \n')

LR_basic = LogisticRegression(train_data.shape[1], 0.001, 10)
LR_basic.train(train_data.to_numpy(), train_labels.to_numpy(), 15)
test_data_basic_predicted_labels_LR = LR_basic.predict(test_data.to_numpy())

print(f'BASIC TEST ACC LR: {accuracy(test_labels, test_data_basic_predicted_labels_LR)} \n')




#Threshold
SVM_threshold = SupportVectorMachine(train_data_threshold.shape[1], 0.001, 10)
SVM_threshold.train(train_data_threshold.to_numpy(), train_labels.to_numpy(), 15)
test_data_threshold_predicted_labels_SVM = SVM_threshold.predict(test_data_threshold.to_numpy())

print('------THRESHOLD: ------ \n')

print(f'THRESHOLD TEST ACC SVM: {accuracy(test_labels, test_data_threshold_predicted_labels_SVM)} \n')

LR_threshold = LogisticRegression(train_data_threshold.shape[1], 0.001, 10)
LR_threshold.train(train_data_threshold.to_numpy(), train_labels.to_numpy(), 15)
test_data_threshold_predicted_labels_LR = LR_threshold.predict(test_data_threshold.to_numpy())

print(f'THRESHOLD TEST ACC LR: {accuracy(test_labels, test_data_threshold_predicted_labels_LR)} \n')




#Variance Threshold
SVM_variance_threshold = SupportVectorMachine(train_data_variance_threshold.shape[1], 0.001, 10)
SVM_variance_threshold.train(train_data_variance_threshold.to_numpy(), train_labels.to_numpy(), 15)
test_data_variance_threshold_predicted_labels_SVM = SVM_variance_threshold.predict(test_data_variance_threshold.to_numpy())

print('------VARIANCE: ------ \n')

print(f'VARIANCE THRESHOLD TEST ACC SVM: {accuracy(test_labels, test_data_variance_threshold_predicted_labels_SVM)} \n')

LR_variance_threshold = LogisticRegression(train_data_variance_threshold.shape[1], 0.001, 10)
LR_variance_threshold.train(train_data_variance_threshold.to_numpy(), train_labels.to_numpy(), 15)
test_data_variance_threshold_predicted_labels_LR = LR_variance_threshold.predict(test_data_variance_threshold.to_numpy())

print(f'VARIANCE THRESHOLD TEST ACC LR: {accuracy(test_labels, test_data_variance_threshold_predicted_labels_LR)} \n')




#Standard Scaler
SVM_standard_scaler = SupportVectorMachine(train_data_standard_scaler.shape[1], 0.001, 10)
SVM_standard_scaler.train(train_data_standard_scaler.to_numpy(), train_labels.to_numpy(), 15)
test_data_standard_scaler_predicted_labels_SVM = SVM_standard_scaler.predict(test_data_standard_scaler.to_numpy())

print('------STANDARD SCALER: ------ \n')

print(f'STANDARD SCALER TEST ACC SVM: {accuracy(test_labels, test_data_standard_scaler_predicted_labels_SVM)} \n')

LR_standard_scaler = LogisticRegression(train_data_standard_scaler.shape[1], 0.001, 10)
LR_standard_scaler.train(train_data_standard_scaler.to_numpy(), train_labels.to_numpy(), 15)
test_data_standard_scaler_predicted_labels_LR = LR_standard_scaler.predict(test_data_standard_scaler.to_numpy())

print(f'STANDARD SCALER TEST ACC LR: {accuracy(test_labels, test_data_standard_scaler_predicted_labels_LR)} \n')




# #Min-Max Scaler








# #PCA 







### ML MODELS ###

# Neural Network
neural_network = MLPClassifier(activation='relu', hidden_layer_sizes=(100,), solver='adam')

neural_network.fit(train_data_threshold, train_labels)

#change these two variable names?
test_data_variance_threshold_predicted_labels_NN = neural_network.predict(test_data_threshold)

print(f'VARIANCE THRESHOLD TEST ACC NN: {accuracy(test_labels, test_data_variance_threshold_predicted_labels_NN)} \n')




#### Pre-Processing: Variance Threshold, PCA, SMOTE, StandardScaler ####

### TESTING FOR SVM and LOGISTIC_REGRESSION #####
vt = VarianceThreshold(threshold=0.05)
pca = PCA(n_components=50)
smote = SMOTE()

X_scaled = standard_scaler.fit_transform(train_data)
X_vt = vt.fit_transform(X_scaled)
X_pca = pca.fit_transform(X_vt)
X_resampled, y_resampled = smote.fit_resample(X_pca, train_labels)

model = SupportVectorMachine(num_features=X_resampled.shape[1], lr0 = 0.01, C=1.0)
model.train(X_resampled, y_resampled, epochs=10)

test_scaled = pca.transform(vt.transform(standard_scaler.transform(test_data)))
y_pred = model.predict(test_scaled)


print(f'ACCURACY FOR SVM: {accuracy(test_labels, y_pred)} \n')


logreg = LogisticRegression(num_features=X_resampled.shape[1], lr0=0.01, sigma2=1.0)
logreg.train(X_resampled, y_resampled, epochs=10)

y_pred_log = logreg.predict(test_scaled)


print(f'ACCURACY FOR LOGREG: {accuracy(test_labels, y_pred_log)} \n')




vt = VarianceThreshold(threshold=0.05)
pca = PCA(n_components=50)
smote = SMOTE()

X_scaled = standard_scaler.fit_transform(train_data)
X_vt = vt.fit_transform(X_scaled)
X_pca = pca.fit_transform(X_vt)


X_resampled, y_resampled = smote.fit_resample(X_pca, train_labels)






X_scaled = StandardScaler().fit_transform(train_data)

X_smote, y_smote = SMOTE().fit_resample(X_scaled, train_labels)

X_final = PCA(n_components=50).fit_transform(X_smote)



rfc = RandomForestClassifier()
rfc.fit(X_vt, train_labels)

x_pred_rfc = rfc.predict(standard_scaler.transform(vt.transform(test_data)))

print(f'Accuracy: {accuracy(test_labels, x_pred_rfc)}')


pipeline = ImbPipeline(steps=[
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=50)),  
    ("smote", SMOTE(random_state=42)),
    ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))
])

param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [None, 10, 20],
    "clf__min_samples_split": [2, 5],
    "clf__class_weight": [None, "balanced", "balanced_subsample"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    verbose=2,
    n_jobs=-1
)

grid.fit(train_data, train_labels)

y_pred = grid.predict(test_data)

print("\nBest Parameters:")
print(grid.best_params_)

print("\nClassification Report:")
print(classification_report(test_labels, y_pred))

print(f"F1 Score: {f1_score(test_labels, y_pred):.4f}")



scaler = StandardScaler()
vt = VarianceThreshold(threshold=0.05)

X_scaled = scaler.fit_transform(train_data)
X_vt = vt.fit_transform(X_scaled)

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_vt, train_labels)

X_train, X_val, y_train, y_val = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=(sum(y_train == 0) / sum(y_train == 1)),
    random_state=42
)

xgb.fit(X_train, y_train)

val_preds = xgb.predict(X_val)
val_f1 = f1_score(y_val, val_preds)

X_eval_scaled = scaler.transform(test_data)
X_eval_vt = vt.transform(X_eval_scaled)
y_pred = xgb.predict(X_eval_vt)
eval_f1 = f1_score(test_labels, y_pred)
report = classification_report(test_labels, y_pred)

val_f1, eval_f1, report


print(val_f1)
print(eval_f1)
print(report)



# Best Parameters:
# {'clf__class_weight': 'balanced_subsample', 'clf__max_depth': 20, 'clf__min_samples_split': 2, 'clf__n_estimators': 200}


#               precision    recall  f1-score   support

#            0       0.00      0.00      0.00         0
#            1       1.00      0.34      0.50      2532

#     accuracy                           0.34      2532
#    macro avg       0.50      0.17      0.25      2532
# weighted avg       1.00      0.34      0.50      2532







########################################################################################################


#### I FOUND THE BEST TO BE: 



# Best Hyperparams: {'lr0': 0.0001, 'reg_tradeoff': 0.1} and best_avg_acc: 0.7203947368421053
# Best Hyperparamters: {'lr0': 0.01, 'reg_tradeoff': 1000} and best avg_acc: 0.7164473684210526


########################################################################################################

# Best Hyperparams: {'lr0': 1, 'reg_tradeoff': 10} and best_avg_acc: 0.33061267815090245
# Best Hyperparamters: {'lr0': 1, 'reg_tradeoff': 1} and best avg_acc: 0.33061267815090245



# X_resampled = pd.DataFrame(X_pca)

# shuffled = X_resampled.sample(frac=1, random_state=42).reset_index(drop=True)

# folds = np.array_split(X_resampled, 5)

# a, b = cross_validation(folds, 'svm', [1, 0.1, 0.01, 0.001, 0.0001], [10, 1, 0.1, 0.01, 0.001], 5)


# print(f'Best Hyperparams: {a} and best_avg_acc: {b}')



# c, d = cross_validation(folds, 'logistic_regression', [1, 0.1, 0.01, 0.001, 0.0001], [1, 10, 100, 1000])





# print(f'Best Hyperparamters: {c} and best avg_acc: {d}')




param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd']
}

grid_search = GridSearchCV(neural_network, param_grid, cv=3)
grid_search.fit(train_data, train_labels)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)


X_train, X_val, y_train, y_val = train_test_split(
    train_data, 
    train_labels, 
    test_size=0.2, 
    random_state=42, 
    stratify=train_labels
)
 







scaler = RobustScaler()
selector = SelectKBest(f_classif, k='all') 

pipeline = ImbPipeline([
    ('scaler', scaler),
    ('variance_threshold', vt),
    ('feature_selection', selector),
    ('resampler', SMOTE(sampling_strategy=0.8, random_state=42)),
    ('classifier', XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    ))
])

param_grid = {
    'feature_selection__k': [50, 100, 'all'],  
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__scale_pos_weight': [1, (sum(y_train == 0) / sum(y_train == 1))]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

val_preds = grid_search.predict(X_val)
print("Validation Set Performance:")
print(classification_report(y_val, val_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_val, val_preds))

if test_data is not None and test_labels is not None:
    test_preds = grid_search.predict(test_data)
    print("\nTest Set Performance:")
    print(classification_report(test_labels, test_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))

best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_model = grid_search.best_estimator_

print("\nBest Parameters:", best_params)
print("Best CV F1 Score:", best_score)











