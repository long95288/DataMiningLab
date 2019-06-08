import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus

class BaggingClassifier():
    def __init__(self, n_model=10):
        '''
        初始化函数
        '''
        #分类器的数量，默认为10
        self.n_model = n_model
        #用于保存模型的列表，训练好分类器后将对象append进去即可
        self.models = []
    def fit(self, feature, label):
        '''
        训练模型，请记得将模型保存至self.models
        :param feature: 训练集数据，类型为ndarray
        :param label: 训练集标签，类型为ndarray
        :return: None
        '''
        #************* Begin ************#
        import random
        from numpy import shape,sum,sign

        def rand_train(feature,label):
            len_train = len(feature)
            train_feature = []
            train_label = []
            for i in range(len_train):
                index = random.randint(0,len_train-1) # 随机生成样本索引
                train_feature.append(feature[index]) # 从所有特征中取出值
                train_label.append(label[index]) # 对应的标签
            return train_feature,train_label # 返回获得的随机训练数据和标签
        
        # 模型
        def bagging_by_tree(feature,label,t=10):
            # predict_list =[]
            for i in range(t):
                train_featrue,train_label = rand_train(feature,label) # 
                clf = DecisionTreeClassifier() # 初始化决策树模型
                clf.fit(train_featrue,train_label) # 训练模型
                # 将模型加入
                self.models.append(clf) # 模型加入
                #total = []
                # label_predict = clf.predict(test_feature) # 预测数据
                # total.append(label_predict)
                # predict_list.append(total)
            
            # 选出
            # return predict_list,label # 返回10个决策树的预测数据
        
        # 投票选出最佳的
        #def vote(predict_list,label):
        #    m,n,k = shape(predict_list)
        #    predict_label = sum(predict_list,axis=0)
        #    predict_label = sign(predict_label)
        #    for i in range(len(predict_label[0])):
        #        if predict_label[0][i] == 0: # 投票数相同,随机生成一个
        #            tip = random.randint(0,1)
        #            if tip == 0:
        #                predict_label[0][i] = 0
        #            else:
        #                predict_label[0][i] = 1
            # 计算
            #error_count = 0 # 错误数量
            #for i in range(k):
            #    if predict_label[0][i] != label[i]:
            #        error_count += 1
        #    return predict_label # 返回预测的参数 
        
        bagging_by_tree(feature,label) # 调用函数生成10个训练模型
        # 对模型进行
        # self.models = vote(predict_list,label) # 使用投票选出预测模型
        
        #from sklearn.cross_validation import train_test_splict
        #from sklearn.ensemble import BaggingClassifier
        # 随机生成训练数据
        #feature_train,feature_test,target_train,target_test = train_test_splict(feature,label,test_size=0.3,random_state=0)
        # tree_models = DecisionTreeClassifier(criterion='gini',max_leaf_nodes=self.n_model)

        #clf = BaggingClassifier(base_es)
        # tree_models.fit(feature,label)
        # self.models = tree_models

        #************* End **************#
    def predict(self, feature):
        '''
        :param feature: 测试集数据，类型为ndarray
        :return: 预测结果，类型为ndarray
        '''
        #************* Begin ************#
        def vote(predict_list): # 投票预测
            predict_list_len = len(predict_list)
            print('组数为',predict_list_len)
            print('每组个数为',len(predict_list[0]))
            print(predict_list[0][111])
            predict_label = np.zeros(len(predict_list[0]),dtype=np.int64)
            for i in range(len(predict_list[0])): # 统计该特征值的位置
                counter_1 = 0 # 值为1的个数
                counter_0 = 0 # 值为0的个i
                for j in range(predict_list_len): # 统计10组数据中的0和1和权重
                    if predict_list[j][i] == 0:
                        counter_0 += 1
                    else:
                        counter_1 += 0
                if counter_0 < counter_1: #1 的票数多
                    predict_label[i] = 1
                elif counter_0 > counter_1: # 0的票数多
                    predict_label[i] = 0
                else: # 1和0的数量一样,随机选一个 
                    random_select = random.randint(0,1)
                    predict_label[i] = random_select
            ##
            print(predict_label)
            return predict_label
        #re = np.array([])
        #for i in range(len(feature)):
        #    re.append(self.models[0][i])
        #return re # 返回预测数据
        # return self.models.predict(feature)
        from numpy import sign,random,sum
        model_num = len(self.models)
        predict_list = [] # 预测结果集
        for i in range(model_num):
            model = self.models[i]
            predict_list.append(model.predict(feature)) # 预测数据
        
        # 对十个模型的预测结果进行投票
        predict_label = vote(predict_list) # 调用投票函数取得最佳的预测
        # predict_label = sum(predict_list,axis=0)
        # predict_label = sign(predict_label)
        #for i in range(len(predict_label)):
        #    if predict_label[i] == 0: # 票数相同选一个
        #        tip = random.randint(0,1)
        #        if tip == 0:
        #            predict_label[i] = 1
        #        else:
        #            predict_label[i] = 0
        # 
        # return self.models[0].predict(feature)
        return predict_label # 返回预测的标签

        #************* End **************#