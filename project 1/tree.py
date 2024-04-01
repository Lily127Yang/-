
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd, numpy as np, csv

from sklearn import model_selection, preprocessing, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, train_test_split
import pandas as pd, numpy as np, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
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
    
    # 使用决策树进行模型训练，添加常见参数
    model = DecisionTreeClassifier(
        criterion='gini',  # 划分标准可以是 "gini" 或 "entropy"
        max_depth= None,    # 决策树的最大深度
        min_samples_split=2,  # 节点划分的最小样本数
        min_samples_leaf=1,   # 叶子节点的最小样本数
        max_features=None,    # 每个节点评估的最大特征数
        random_state=42       # 随机种子，用于控制随机性
    )
    
    print('this model is running')
    starttime = datetime.datetime.now()
    model.fit(X_train, y_train)
    endtime = datetime.datetime.now()
    print('this model finishes running')
    print('this model finishes running, running time:', (endtime - starttime).seconds, 'seconds.')
    score = model.score(X_val, y_val)
    print("Model score:", score)

    # 计算模型的准确率
    accuracy = accuracy_score(y_val, model.predict(X_val))
    print("Model Accuracy:", accuracy)

    # 计算模型的精确率
    precision = precision_score(y_val, model.predict(X_val), average='macro')
    print("Model Precision:", precision)

    # 计算模型的召回率
    recall = recall_score(y_val, model.predict(X_val), average='macro')
    print("Model Recall:", recall)

    # 计算模型的F1分数
    f1 = f1_score(y_val, model.predict(X_val), average='macro')
    print("Model F1 Score:", f1)


    # 对测试集进行预测
    dataset_test = pd.read_csv('test.csv')
    X_test = dataset_test['text']
    X_test = X_test.values.tolist()
    
    X_test_fit = tv.transform(X_test).toarray()
    print(X_test_fit.shape)

    y_pred = model.predict(X_test_fit)
    y_pred = np.expand_dims(y_pred, axis=1)
    
    X_test_id = dataset_test['id']
    X_test_id = X_test_id.to_numpy()
    X_test_id = np.expand_dims(X_test_id, axis=1)
    
    result = np.concatenate((X_test_id, y_pred), axis=1)
    label = np.array([['id', 'pred']])
    result = np.concatenate((label, result), axis=0)
    
    np.savetxt('results_DecisionTree.txt', result, delimiter=', ', fmt='%s')
