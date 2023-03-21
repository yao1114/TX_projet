import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score

# Define a function to read and split the dataset
def read_and_split_data(file_path, test_size=0.2, random_state=40):
    data = pd.read_csv(file_path)
    # Copy target variable
    y = data['session'].copy()
    # Replace 1 and 3 with 2, and 2 with 1 in y
    y.replace({1: 2, 2: 1, 3: 2}, inplace=True)
    # Split the dataset into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], y, test_size=test_size, 
                                                        stratify=y, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Define a function to evaluate model performance
def evaluate_model_performance(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
    print('AUC: ', roc_auc_score(y_test, y_pred))
    print('F1-score: ', f1_score(y_test, y_pred))

# Define a pipeline to link data preprocessing and model training and evaluation
pipeline = Pipeline([
    ('classifier', XGBClassifier(random_state=40))
])

# Define hyperparameter space
param_grid = {
    'classifier__max_depth': [2, 5, 10, 15],
    'classifier__min_child_weight': [1, 5, 10],
    'classifier__gamma': [0.5, 1, 1.5, 2, 5],
    'classifier__subsample': [0.6, 0.8, 1.0],
    'classifier__colsample_bytree': [0.6, 0.8, 1.0],
    'classifier__reg_alpha': [0, 0.001, 0.005, 0.01, 0.05],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.15]
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5, # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,
    error_score='raise' # Raise an exception if a fit fails
)

# Read and split the dataset
X_train, X_test, y_train, y_test = read_and_split_data('HRV_ECG_step60.csv')

# Fit GridSearchCV object on the training dataset
grid_search.fit(X_train, y_train)

# Print best hyperparameters
print('Best parameters:', grid_search.best_params_)

# Fit XGBClassifier object with best hyperparameters on the training dataset
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Evaluate model performance
evaluate_model_performance(best_model, X_test, y_test)