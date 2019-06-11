import numpy as np
from collections import  Counter
from sklearn.tree import DecisionTreeClassifier
class RandomForestClassifier():
    def __init__(self, n_model=10):
        '''
        初始化函数
        '''
        #分类器的数量，默认为10
        self.n_model = n_model
        #用于保存模型的列表，训练好分类器后将对象append进去即可
        self.models = []
        #用于保存决策树训练时随机选取的列的索引
        self.col_indexs = []
    def fit(self, feature, label):
        '''
        训练模型
        :param feature: 训练集数据，类型为ndarray
        :param label: 训练集标签，类型为ndarray
        :return: None
        '''
        #************* Begin ************#
        import random
        # 有放回的采样数据函数
        def get_random_simple_label(feature,label):
            len_train = len(feature)
            train_feature = []
            train_label = []
            for i in range(len_train):
                index = random.randint(0,len_train-1) # 随机生成样本索引
                train_feature.append(feature[index]) # 从所有特征中取出值
                train_label.append(label[index]) # 对应的标签
            return train_feature,train_label # 返回获得的随机训练数据和标签
        #################################################

        # 获得随机特征训练数据函数
        def get_rand_train_data(feature):
            feature_num = len(feature[0]) # 特征的个数
            indexlist = range(feature_num) # 列的索引
            k =int(np.log2(feature_num)) # 随机选取特征的个k
            print('选取特征值',k)
            random_index = random.sample(indexlist,k) # 随机选择log2(n)个特征值
            # random_index.sort() # 排序便于选择
            print(random_index)
            # 获得feature的子集
            train_feature = []
            print('feature的组数',len(feature))
            feature_len = len(feature)
            for i in range(feature_len):
                temp_array = []
                for j in random_index:
                    temp_array.append(feature[i][j]) #获得值
                # print(temp_array)
                train_feature.append(temp_array)
            #pass
            print('训练集组数',len(train_feature)) # 长度和feature长度一样的才对
            return train_feature,random_index # 返回子集和子集对应的索引
        #######################################################

        # 构建随机森林
        for i in range(self.n_model):
            train_simple,train_label = get_random_simple_label(feature,label)
            train_feature,col_index = get_rand_train_data(train_simple)
            # 构造树
            clf = DecisionTreeClassifier()
            clf.fit(train_feature,train_label)
            self.models.append(clf) # 加入模型组
            self.col_indexs.append(col_index) # 加入模型对应的特征组的索引
        
        #clf = DecisionTreeClassifier()
        #clf.fit(feature,label)
        #self.models.append(clf)
        #get_rand_train_data(feature)

        #************* End **************#
    def predict(self, feature):
        '''
        :param feature: 测试集数据，类型为ndarray
        :return:预测结果，类型为ndarray，如np.array([0, 1, 2, 2, 1, 0])
        '''
        #************* Begin ************#

        # 获得索引对应的特征数据
        def get_predict_feature(feature,index_list):
            predict_feature = []
            for i in range(len(feature)):
                temp_feature = []
                for j in index_list:
                    temp_feature.append(feature[i][j])
                predict_feature.append(temp_feature)

            return predict_feature # 返回索引对应的特征值
        #########################################
        # 投票函数
        import random
        def vote(predict_list): # 投票预测
            predict_list_len = len(predict_list)
            print('组数为',predict_list_len)
            print('每组个数为',len(predict_list[0]))
            print(predict_list[0][111])
            predict_label = np.zeros(len(predict_list[0]),dtype=np.int64)
            for i in range(len(predict_list[0])): # 统计该特征值的位置
                counter_1 = 0 # 值为1的个数
                counter_0 = 0 # 值为0的个i
                counter_2 = 0 # 值为2的个数
                for j in range(predict_list_len): # 统计10组数据中的0和1和权重
                    if predict_list[j][i] == 0:
                        counter_0 += 1
                    elif predict_list[j][i] == 1:
                        counter_1 += 1
                    else:
                        counter_2 += 1
                
                if counter_0 < counter_1 and counter_2 < counter_1: # 1 的票数多
                    predict_label[i] = 1
                elif counter_0 > counter_1 and counter_0 > counter_2: # 0的票数多
                    predict_label[i] = 0
                elif counter_2 > counter_1 and counter_2 > counter_0: # 2的票数多
                    predict_label[i] = 2
                else: # 1和0的数量一样,随机选一个 
                    random_select = random.randint(0,2)
                    predict_label[i] = random_select
            ##
            print(predict_label)
            return predict_label
        ########################################
        model_num = len(self.models)
        predict_list = []
        print('模型数',model_num)
        print('索引数',len(self.col_indexs))
        for i in range(model_num):
            model = self.models[i]
            # 获得该模型对应的特征值测试数据
            index = self.col_indexs[i]
            predict_feature = get_predict_feature(feature,index)
            predict_list.append(model.predict(predict_feature)) # 预测数据
        
        predict_label = vote(predict_list) # 对预测数据进行投票
        print('预测结果',predict_label)
        return predict_label # 返回预测标签值
        #************* End **************#