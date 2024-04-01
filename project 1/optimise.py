from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import load_files
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, train_test_split
import pandas as pd, numpy as np, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

import csv
import datetime
import pickle

# 由于 train_data.txt 与 test.txt 的文件内容组织形式不一样，所以我们需要分别将其转化为统一的 csv 格式
def txt_to_csv_train(filePathSrc, filePathDst):
    list_data = []
    with open(filePathSrc, 'r', encoding='utf-8') as input_file, open(filePathDst, 'w', newline='') as output_file:
        for data in input_file:
            data = eval(data)
            list_data.append(data)
        keys = list_data[0].keys()
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(list_data)

def txt_to_csv_test(filePathSrc, filePathDst):
    with open(filePathSrc, 'r', encoding='utf-8') as input_file, open(filePathDst, 'w', newline='') as output_file:
        stripped = (line.strip('\n') for line in input_file)
        # 这里只需要用第一个逗号进行分割，因为一个句子中可能有很多个逗号
        lines = (line.split(', ', 1) for line in stripped if line)
        writer = csv.writer(output_file)
        writer.writerows(lines)



if __name__ == '__main__':
    txt_to_csv_train('train_data.txt', 'train_data.csv')
    txt_to_csv_test('test.txt', 'test.csv')

    dataset_train = pd.read_csv('train_data.csv')
    X = dataset_train.drop('label', axis=1)
    y = dataset_train['label']
    X = X.values.tolist()
    list1 = []
    for index_i in range(len(X)):
        for index_j in range(len(X[index_i])):
            list1.append(X[index_i][index_j])
    tv = TfidfVectorizer(stop_words='english')
    X_fit = tv.fit_transform(list1).toarray()
    print(X_fit.shape)

# 数据集划分，进行模型的训练以及预测
X_train, X_val, y_train, y_val = train_test_split(X_fit, y, test_size=0.2, random_state=42)
# 创建决策树模型
decision_tree = DecisionTreeClassifier()
# 定义要优化的参数范围
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}
# 创建GridSearchCV对象
grid_search = GridSearchCV(decision_tree, param_grid, cv=4)
print('start selecting...')
# 执行网格搜索
grid_search.fit(X_fit, y)
# 输出最佳参数和对应的评分
print("The best parameters are %s with a score of %0.2f" %(grid_search.best_params_, grid_search.best_score_))
