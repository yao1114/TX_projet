# 导入所需的库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score
from sklearn.decomposition import PCA


pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

# Read dataset
data = pd.read_csv('HRV_ECG_step60.csv')
column_headers = list(data.columns.values)

# 复制y值
y_binary = data['session'].copy()

# 将y_binary中的1和3替换为2，2替换为1
y_binary.replace({1: 2, 2: 1, 3: 2}, inplace=True)

# 标准化自变量数据
X = data.iloc[:, :-1]
X = (X - X.mean()) / X.std()

# 使用PCA将自变量数据转换为主成分，并减少其数量
pca = PCA(n_components=0.95) # 保留95%的方差
X_pca = pca.fit_transform(X)

# Dividing the data set into a training set and a test set
unique_ids = data['ID'].unique()

test_ids = set(pd.Series(unique_ids).sample(frac=0.2, random_state=40))

train_data = data[data['ID'].apply(lambda x: x not in test_ids)]
test_data = data[data['ID'].apply(lambda x: x in test_ids)]

# 使用y_binary作为训练和测试数据集中的y值
y_train = y_binary[train_data.index]
y_test = y_binary[test_data.index]

# 使用PCA转换后的主成分作为新的自变量数据
X_train_pca = X_pca[train_data.index]
X_test_pca = X_pca[test_data.index]

# --------------- --------Training------- ---------------
# Decision tree classifier
classifier = DecisionTreeClassifier()

# Training models
classifier.fit(X_train_pca, y_train)

# Predictive test sets
y_pred = classifier.predict(X_test_pca)

# Evaluation models
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
print('AUC: ', roc_auc_score(y_test, y_pred))
print('F1-score: ', f1_score(y_test, y_pred))