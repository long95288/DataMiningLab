import pandas as pd

from RandomForestClassifier import RandomForestClassifier

#读入输入数据
#train = pd.read_csv('../data/train_data.csv', header=0)
#test = pd.read_csv('../data/test_data.csv', header=0)
train = pd.read_csv('../data/soybean_train.csv',header=0)
test = pd.read_csv('../data/soybean_test.csv',header=0)

### 对数据进行预处理 ###
#train_pf = pd.DataFrame(train)
#train_pf.duplicated() # 去除重复值
#print(train_pf.info())
def dd(train,test):
    # 对数据进行预处理
    train_columns_size = train.columns.size
    test_columns_size = test.columns.size
    
    for i in range(train_columns_size):
        row_len = len(train[:,0:1])
        print('行数',row_len)
    return train,test
###############################
columns = train.columns.size
dd(train,test)
# train2 = train.copy() 
# for i in range(columns):
#      tmp_data = train.values[:,i-1:i]
#      # mode = tmp_data.mode() # 中位数
#      # print('中位数',mode)
#      for j in range(len(tmp_data)):
#          if tmp_data[j] == '?':
#             # tmp_data[j] = '' # 将?替换成空值，以便于后续处理
#             print('列数%d,行数%d',i-1,j+1)
#             train2.values[j+1][i-1] =None # 置空
#     #train[i] = tmp_data
# train = train2
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