import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score
from sklearn.decomposition import FastICA

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

# Read dataset
data = pd.read_csv('HRV_ECG_step60.csv')
column_headers = list(data.columns.values)

y_binary = data['session'].copy()

y_binary.replace({1: 2, 2: 1, 3: 2}, inplace=True)

# Original
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(data.iloc[:, :-1], y_binary, test_size=0.2, 
                                                            stratify=y_binary, random_state=40)

# Divided with same IDs split together
unique_ids = data['ID'].unique()
test_ids = set(pd.Series(unique_ids).sample(frac=0.2, random_state=40))
train_data = data[data['ID'].apply(lambda x: x not in test_ids)]
test_data = data[data['ID'].apply(lambda x: x in test_ids)]
X_train, y_train = train_data.iloc[:, :-1], y_binary[train_data.index]
X_test, y_test = test_data.iloc[:, :-1], y_binary[test_data.index]

# Divided by proportion
ID = data['ID'].copy()
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(data.iloc[:, :-1], y_binary, test_size=0.2, 
                                                            stratify=ID, random_state=40)
# --------------- --------Training------- ---------------
# Decision tree classifier
classifier = DecisionTreeClassifier()

# Training models: Original
classifier.fit(X_train_o, y_train_o)

# Predictive test sets
y_pred_o = classifier.predict(X_test_o)


# Evaluation models
print('-------------------Original-------------------')
print('Accuracy: ', accuracy_score(y_test_o, y_pred_o))
print('Confusion Matrix: \n', confusion_matrix(y_test_o, y_pred_o))
print('AUC: ', roc_auc_score(y_test_o, y_pred_o))
print('F1-score: ', f1_score(y_test_o, y_pred_o))

# Training models: Divided with same IDs split together
classifier.fit(X_train, y_train)

# Predictive test sets
y_pred = classifier.predict(X_test)

# Evaluation models
print('-------------------Divided with same IDs split together-------------------')
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
print('AUC: ', roc_auc_score(y_test, y_pred))
print('F1-score: ', f1_score(y_test, y_pred))

# Training models: Divided by proportion
classifier.fit(X_train_p, y_train_p)

# Predictive test sets
y_pred_p = classifier.predict(X_test_p)

# Evaluation models
print('-------------------Divided by proportion-------------------')
print('Accuracy: ', accuracy_score(y_test_p, y_pred_p))
print('Confusion Matrix: \n', confusion_matrix(y_test_p, y_pred_p))
print('AUC: ', roc_auc_score(y_test_p, y_pred_p))
print('F1-score: ', f1_score(y_test_p, y_pred_p))

