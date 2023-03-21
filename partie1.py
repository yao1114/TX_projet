import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 以读方式打开文件
df = pd.read_csv("HRV_ECG_step60_mona.csv")

# Delete 'nni_histogram', 'nni_diff_mean', 'sdnn_index', 'sdann'
df=df.drop(['nni_histogram', 'nni_diff_mean', 'sdnn_index', 'sdann'],axis=1)

column_headers = list(df.columns.values)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# --------------- --------PCA------- ---------------
# PCA
n_pca = 40
pca = PCA(n_components=n_pca)
X_pca = pca.fit_transform(X)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Print the corresponding PC% for each feature
variance_ratios = pca.explained_variance_ratio_
variance_dict = dict(zip(column_headers, variance_ratios))
print("Feature\t\tVariance Ratio")
print("-------\t\t--------------")
for feature, ratio in variance_dict.items():
    print("{:<15}\t{:.10f}".format(feature, ratio))

# Calculate the total value of the top 50 PC
cumulative_variances = np.cumsum(variance_ratios)
total_variance = cumulative_variances[n_pca-1]
print("Total variance explained by the first 50 components: {:.10f}%".format(total_variance * 100))


# --------------- --------IQR = Q3 − Q1------- ---------------

# 确定要进行异常值检测的特征列
columns_to_check = ['nni_counter', 'nni_mean', 'nni_min']

# 计算每个特征列的均值和标准差
means = df[columns_to_check].mean()
stds = df[columns_to_check].std()

# 使用均值和标准差计算上下界
lower_bounds = means - 3 * stds
upper_bounds = means + 3 * stds

# 检测异常值
outliers = []
for column in columns_to_check:
    lower_bound = lower_bounds[column]
    upper_bound = upper_bounds[column]
    column_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    outliers.append(column_outliers)
outliers = pd.concat(outliers)

# 删除异常值
df = df.drop(outliers.index)



df.to_csv('HRV_ECG_step60.csv', index=False)