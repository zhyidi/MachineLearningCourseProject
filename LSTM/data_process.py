# -*- coding: utf-8 -*- 
# @Time : 2023/12/6 16:38 
# @Author : lf_Liu 
# @File : data_process.py
import torch
import numpy as np
import pandas as pd
from config import parse_args
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
np.random.seed(0)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def np2tensor(data):
    x_tensor = torch.from_numpy(data[0]).to(torch.float32)
    y_tensor = torch.from_numpy(data[1]).to(torch.float32)
    return x_tensor,y_tensor

def process_timestep(data,x_timestep,y_timestep):
    dataX = []  # 保存X
    dataY = []  # 保存Y
    # 将整个x_timestep长的数据保存到X中，将未来y_timestep天保存到Y中
    for index in range(len(data) - x_timestep - y_timestep+1):
        dataX.append(data[index:index + x_timestep])
        dataY.append(data[index + x_timestep:index + x_timestep + y_timestep])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    return [dataX,dataY]

def data_processing(arg):
    # raw_data = pd.read_csv(arg.data_path, encoding='utf-8')
    x_timestep, y_timestep = arg.x_timestep,arg.y_timestep
    # 不存在缺失值
    # print(raw_data.isnull().any())
    # 1、划分训练集 验证集 测试集
    # 划分训练集、验证集、测试集 6:2:2
    data_train = pd.read_csv(arg.data_path_train, encoding='utf-8').iloc[:,1:]
    data_valid = pd.read_csv(arg.data_path_valid, encoding='utf-8').iloc[:,1:]
    data_test = pd.read_csv(arg.data_path_test, encoding='utf-8').iloc[:,1:]

    # 2、
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = StandardScaler()
    scaler.fit(data_train)
    data_train_norm = scaler.transform(data_train)
    data_valid_norm = scaler.transform(data_valid)
    data_test_norm = scaler.transform(data_test)
    # 反归一化
    # scaler.inverse_transform(data_test_norm)

    # 3、根据时间片划分数据
    data_train_step = process_timestep(data_train_norm, x_timestep, y_timestep )
    data_valid_step = process_timestep(data_valid_norm, x_timestep, y_timestep )
    data_test_step = process_timestep(data_test_norm, x_timestep, y_timestep )

    x_train_tensor,y_train_tensor = np2tensor(data_train_step)
    x_valid_tensor,y_valid_tensor = np2tensor(data_valid_step)
    x_test_tensor,y_test_tensor = np2tensor(data_test_step)

    # 形成数据集
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    valid_data = TensorDataset(x_valid_tensor, y_valid_tensor)
    test_data = TensorDataset(x_test_tensor, y_test_tensor)

    # 将数据加载成迭代器
    # 参数为True随机打乱 训练集验证集均随机打乱
    train_loader = torch.utils.data.DataLoader(train_data,
                                               arg.train_batch_size,
                                               arg.shuffle)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                              arg.valid_batch_size,
                                              arg.shuffle)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              arg.test_batch_size,
                                              False)
    return scaler,train_loader,valid_loader,test_loader,(train_data,valid_data,test_data)

if __name__ == '__main__':

    arg = parse_args()
    scaler,train_loader,dev_loader,test_loader,test_data = data_processing(arg)

