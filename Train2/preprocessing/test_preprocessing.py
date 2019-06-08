import pandas as pd
from preprocessing import preprocessing

#读入输入数据
data = pd.read_csv('../data/audit_risk.csv', header=0)

#对数据预处理
processed_data = preprocessing(data)

#将预处理后的数据写入csv文件
save = pd.DataFrame(processed_data)
save.to_csv('../result/result_preprocessing.csv', index=False, header=True)

print('True', end='')