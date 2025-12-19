from utils import accuracy

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from imblearn.pipeline import pipeline
from sklearn.pipeline import Pipeline

import pandas as pd



#### Load Data ####
train_data = pd.read_csv(r"D:\STAT 6969\project\project_data\data\train.csv") # <--- CHANGE THIS to fit your stored location for this file
test_data = pd.read_csv(r"D:\STAT 6969\project\project_data\data\test.csv") # <--- CHANGE THIS to fit your stored location for this file

train_labels = train_data['label']
test_labels = test_data['label']

train_data = train_data.drop(columns = 'label')
test_data = test_data.drop(columns = 'label')

eval_ids = pd.read_csv(r'D:\CS 6350\project_data\data\eval.id', header=None, names = ['example_id'])




#### Instantiate Model ####
log_reg_model = LogisticRegression(max_iter = 1000)


#### Model without Preprocessing ####
log_reg_model.fit(train_data, train_labels)

train_preds = log_reg_model.predict(train_data)
test_preds = log_reg_model.predict(test_data)

print("=== Without Preprocessing ===")
print(f"Train Accuracy: {accuracy_score(train_labels, train_preds)}")
print(f"Train F1 Score: {f1_score(train_labels, train_preds)}")
print(f"Test Accuracy: {accuracy_score(test_labels, test_preds)}")
print(f"Test F1 Score: {f1_score(test_labels, test_preds)}")


#### Model WITH Preprocessing ####
pipeline = Pipeline([
    ('var_thresh', VarianceThreshold(threshold=0.0)),
    ('scaler', StandardScaler()),                    
    ('logreg', LogisticRegression(max_iter=1000))  
])

pipeline.fit(train_data, train_labels)

train_preds_pre = pipeline.predict(train_data)
test_preds_pre = pipeline.predict(test_data)

print("\n=== With Preprocessing ===")
print(f"Train Accuracy: {accuracy_score(train_labels, train_preds_pre)}")
print(f"Train F1 Score: {f1_score(train_labels, train_preds_pre)}")
print(f"Test Accuracy: {accuracy_score(test_labels, test_preds_pre)}")
print(f"Test F1 Score: {f1_score(test_labels, test_preds_pre)}")


# === Without Preprocessing ===
# Train Accuracy: 0.7899170725286297
# Train F1 Score: 0.692485549132948
# Test Accuracy: 0.7933623073883841
# Test F1 Score: 0.7050197405527355

# === With Preprocessing ===
# Train Accuracy: 0.820192181124128
# Train F1 Score: 0.7550215208034433
# Test Accuracy: 0.8186487554326353
# Test F1 Score: 0.7559808612440191

print("Conclusion: \nIn my run, test accuracy improved by about 2 percent and the F1 Score was up about 5, so the pre-processing did help! (see above in the code)")