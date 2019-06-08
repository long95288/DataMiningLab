import pandas as pd

from RandomForestClassifier import RandomForestClassifier

#读入输入数据
train = pd.read_csv('../data/train_data.csv', header=0)
test = pd.read_csv('../data/test_data.csv', header=0)

columns = train.columns.size
train_data = train.values[:, 1:columns-1]
train_labels = train.values[:, columns-1]
test_data = test.values[:, 1:columns-1]
test_labels = test.values[:, columns-1]

#调用RandomForest方法
bagging = RandomForestClassifier(10)
bagging.fit(train_data, train_labels)
predicted_labels = bagging.predict(test_data)

correct = 0
for i in range(len(test_data)):
    if(predicted_labels[i] == test_labels[i]):
        correct += 1

#输出结果
correct_ratio = correct / len(test_data)
if(correct_ratio > 0.8):
    print("正确率超过0.8,为",correct_ratio, end='')
else:
    print("正确率为：", correct_ratio, end='')