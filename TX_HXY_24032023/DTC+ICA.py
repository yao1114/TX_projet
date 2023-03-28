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

# Dividing the data set into a training set and a test set
unique_ids = data['ID'].unique()

test_ids = set(pd.Series(unique_ids).sample(frac=0.2, random_state=40))

train_data = data[data['ID'].apply(lambda x: x not in test_ids)]
test_data = data[data['ID'].apply(lambda x: x in test_ids)]

X_train, y_train = train_data.iloc[:, :-1], y_binary[train_data.index]
X_test, y_test = test_data.iloc[:, :-1], y_binary[test_data.index]

# FastICA
ica = FastICA(n_components=10)
X_train = pd.DataFrame(ica.fit_transform(X_train))
X_test = pd.DataFrame(ica.transform(X_test))

# --------------- --------Training------- ---------------
# Decision tree classifier
classifier = DecisionTreeClassifier()

# Training models
classifier.fit(X_train, y_train)

# Predictive test sets
y_pred = classifier.predict(X_test)

# Evaluation models
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
print('AUC: ', roc_auc_score(y_test, y_pred))
print('F1-score: ', f1_score(y_test, y_pred))