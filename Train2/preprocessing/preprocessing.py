# -*- coding: utf-8 -*-
def preprocessing(data):
    """对data进行预处理
    参数:
        data - pandas.dataframe格式，原始数据

    返回值：
        processed_data - 预处理后的数据
    """
    processed_data = data
    #   请在此添加实现代码     #
    #********** Begin *********#
    # print(data)
    # 删除拥有缺失值的行
    processed_data.dropna(axis=0,how='any',inplace=True)
    # 删除重复的数据
    #for column in data.columns:
    #    column_data =  data[column] # 某列的数据
        # 对该列进行缺失值删除
    #    for i in range(len(column_data)):
    #        if column_data[i] is None:
                # 删除该index 对应的行
    #            processed_data.drop(processed_data.index[i],inplace=True)
    # 处理Sector_score字段 最大值为 最小值为
    min = data['Sector_score'].min
    max = data['Sector_score'].max
    #print('最小值',min)
    # print('最大值',max)
    # 处理Location_id 将字符串的行删除
    print(len(processed_data))
    location_id = processed_data['LOCATION_ID'] # 字段值
    id_len = len(location_id) # 字段长度
    print('location_id长度',id_len)
    for i in range(id_len):
        if location_id[i].isalpha():
            print('索引',i) 
    #
    # Detection_Risk字段都是0.5唯一值，不具有分析价值，删除掉
    del processed_data['Detection_Risk']
    
    #********** End ***********#
    return processed_data