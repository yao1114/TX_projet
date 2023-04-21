import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score

# Define a function to read and split the dataset
def read_and_split_data_(file_path, test_size=0.2, random_state=40):
    data = pd.read_csv(file_path)
    # Copy target variable
    y = data['session'].copy()
    # Replace 1 and 3 with 2, and 2 with 1 in y
    y.replace({1: 2, 2: 1, 3: 2}, inplace=True)
    # Split the dataset into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], y, test_size=test_size, 
                                                        stratify=y, random_state=random_state)
    return X_train, X_test, y_train, y_test

def read_and_split_data(file_path, test_size=0.2, random_state=40):
    data = pd.read_csv(file_path)

    y_binary = data['session'].copy()
    y_binary.replace({1: 2, 2: 1, 3: 2}, inplace=True)

    # Divided with same IDs split together
    unique_ids = data['ID'].unique()
    test_ids = set(pd.Series(unique_ids).sample(frac=0.2, random_state=40))
    train_data = data[data['ID'].apply(lambda x: x not in test_ids)]
    test_data = data[data['ID'].apply(lambda x: x in test_ids)]
    X_train, y_train = train_data.iloc[:, :-1], y_binary[train_data.index]
    X_test, y_test = test_data.iloc[:, :-1], y_binary[test_data.index]

    return X_train, X_test, y_train, y_test


# Define a function to evaluate model performance
def evaluate_model_performance(model, X_test, y_test):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted', labels=[1, 2])

    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print('Accuracy for class Stress: {:.3f}'.format(accuracy_score(y_test[y_test==1], y_pred[y_test==1])))
    print('Accuracy for class Relax: {:.3f}'.format(accuracy_score(y_test[y_test==2], y_pred[y_test==2])))
    print('\nConfusion Matrix: \n', confusion_matrix(y_test, y_pred, normalize='true'))
    print('\nAUC: ', roc_auc_score(y_test, y_pred))
    print('\nF1-score for class Stress: {:.3f}'.format(f1_score(y_test[y_test==1], y_pred[y_test==1], average='weighted')))
    print('F1-score for class Relax: {:.3f}'.format(f1_score(y_test[y_test==2], y_pred[y_test==2], average='weighted')))
    print('F1-score:', f1_score(y_test, y_pred))


# Define a pipeline to link data preprocessing and model training and evaluation
pipeline = Pipeline([
    ('classifier', RandomForestClassifier(random_state=40))
])

# Define hyperparameter space
param_grid = {
    'classifier__n_estimators': [100, 300, 500],
    'classifier__max_depth': [2, 5, 10, 15],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=LeaveOneOut(), # LeaveOneOut
    scoring='accuracy',
    n_jobs=-1,
)

# Read and split the dataset
X_train, X_test, y_train, y_test = read_and_split_data('HRV_ECG_step60.csv')

# Fit GridSearchCV object on the training dataset
grid_search.fit(X_train, y_train)

# Print best hyperparameters
print('Best parameters:', grid_search.best_params_)

# Fit RandomForestClassifier object with best hyperparameters on the training dataset
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Evaluate model performance
evaluate_model_performance(best_model, X_test, y_test)