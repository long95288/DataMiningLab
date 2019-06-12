# coding=utf-8
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

# 相关函数
# 单列填充中位数
def fill_Median(data):
    processed_data = data
    processed_data = data.fillna(data.median())
    return processed_data
###############################
# 读入未处理的 还有?号的
raw_train_data = pd.read_csv('data/soybean-large.data',header=None)
raw_test_data = pd.read_csv('data/soybean-large.test',header=None)

# 1、将?置空 NAN
raw_train_data.replace('?',np.nan,inplace=True)
raw_test_data.replace('?',np.nan,inplace=True)
# 2、将NAN改成列的中位数
train_columns = raw_train_data.columns.size
i = 1
while i < train_columns: # 处理空值,
    raw_train_data[i] = fill_Median(raw_train_data[i])
    i += 1
# print(raw_train_data)
train = raw_train_data
train_data = train.values[:, 1:train_columns]
train_labels = train.values[:, 0:1]

test = raw_test_data
test_columns = test.columns.size
i = 1
while i < test_columns: # 处理空值
    raw_test_data[i] = fill_Median(raw_test_data[i])
    i += 1

test_data = test.values[:, 1:test_columns]
test_labels = test.values[:, 0:1]

#建立模型
#clf = DecisionTreeClassifier()
#clf.fit(train_feat, train_labels)
# 
# 随机森林
forest_train_feature,forest_train_labels = shuffle(train_data,train_labels,random_state=3)
clf = RandomForestClassifier()
clf.fit(forest_train_feature,forest_train_labels.ravel())

predicted_labels = clf.predict(test_data)


correct = 0
for i in range(len(test_data)):
    if(predicted_labels[i] == test_labels[i]):
        correct += 1

#输出结果
correct_ratio = correct / len(test_data)
if(correct_ratio > 0.8):
    print("正确率超过0.8", end='')
    print('正确率为',correct_ratio)
else:
    print("正确率为：", correct_ratio, end='')

# 写入文件中
f = open('data/predict_labels.csv','a')
f.writelines(predicted_labels.ravel())
