import pandas as pd
import numpy as np
#from RandomForestClassifier import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#读入输入数据
#train = pd.read_csv('../data/train_data.csv', header=0)
#test = pd.read_csv('../data/test_data.csv', header=0)
train = pd.read_csv('../data/soybean_train.csv',header=0)
test = pd.read_csv('../data/soybean_test.csv',header=0)

### 对数据进行预处理 ###
train_pf = pd.DataFrame(train)
print(train_pf.shape)
#train_pf.duplicated() # 去除重复值
#print(train_pf.info())
def fill_median(data): # 对?进行处理
    # 对数据进行预处理
    processed_data = data
    median_num = '1'
    len_data = len(processed_data)
    #
    temp_data = processed_data.copy()
    for i in range(len_data):
        if processed_data[i] == '?':
            temp_data[i] = median_num # 中位数替换?
    processed_data = temp_data
    return processed_data
###############################
# 将第一列序列化
def strTonum(data):
    temp_data = np.zeros((len(data),1),dtype=int) # 建立一个临时变量
    types = ['diaporthe-stem-canker',
             'charcoal-rot',
             'rhizoctonia-root-rot',
             'phytophthora-rot',
             'brown-stem-rot',
             'powdery-mildew',
             'downy-mildew',
             'brown-spot',
             'bacterial-blight',
             'bacterial-pustule',
             'purple-seed-stain',
             'anthracnose',
             'phyllosticta-leaf-spot',
             'alternarialeaf-spot',
             'frog-eye-leaf-spot',
             'diaporthe-pod-&-stem-blight',
             'cyst-nematode',
             'herbicide-injury',
             '2-4-d-injury']
    for i in range(len(data)):
        flag = False # 是否已经赋值标记
        for j in range(len(types)):
            if data[i] == types[j]:
                # 赋值编号
                temp_data[i] = j
                flag = True
        if flag != True:
            print('未知数据')
            temp_data[i] = len(temp_data) # 其它类别
    return temp_data

# 处理训练集
columns = train.columns.size
raw_train_data = np.array(train.values[:,0:1]) # 第一列不用处理 
# 处理第一列 数字化
raw_train_data = strTonum(raw_train_data) 
print('序列化之后')
print(raw_train_data)

for i in range(columns):
    if i >0:
        temp_data = fill_median(train.values[:,i:i+1])
        raw_train_data = np.hstack((raw_train_data,temp_data)) # 合并数据
        
print(raw_train_data.shape)

train_data = raw_train_data[:, 1:columns]
train_labels = raw_train_data[:, 0:1]
print('训练标签',train_labels)
# 处理测试集
test_columns = test.columns.size

raw_test_data = np.array(test.values[:,0:1])
raw_test_data = strTonum(raw_test_data)
for i in range(test_columns):
    if i > 0:
        temp_data = fill_median(test.values[:,i:i+1])
        raw_test_data = np.hstack((raw_test_data,temp_data))

# 测试数据维数
print('测试数据维数',raw_test_data.shape)
test_data = raw_test_data[:,1:test_columns]
test_labels = raw_test_data[:,0:1]

#调用RandomForest方法
# bagging = RandomForestClassifier(10)
# bagging.fit(train_data, train_labels)
# predicted_labels = bagging.predict(test_data)
clf = DecisionTreeClassifier()
clf.fit(train_data,train_labels)
predicted_labels = clf.predict(test_labels)

print('预测集')
print(predicted_labels)
print('测试集标签')
print(test_labels)

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