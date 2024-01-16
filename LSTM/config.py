# -*- coding: utf-8 -*- 
# @Time : 2023/12/7 20:23 
# @Author : lf_Liu 
# @File : config.py
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Time Series forecast')
    parser.add_argument('-model', type=str, default='LSTM', help="")
    parser.add_argument('-x_timestep', type=int, default=96, help="时间窗口大小")
    # 64/336
    parser.add_argument('-y_timestep', type=int, default=96, help="预测未来数据长度")
    # data
    parser.add_argument('-shuffle', action='store_true', default=True, help="是否打乱数据加载器中的数据顺序")
    parser.add_argument('-data_path_train', type=str, default='train_set.csv', help="")
    parser.add_argument('-data_path_valid', type=str, default='validation_set.csv', help="")
    parser.add_argument('-data_path_test', type=str, default='test_set.csv', help="")

    # learning
    # 1e-5可能太小了
    parser.add_argument('-lr', type=float, default=5e-4, help="学习率")
    parser.add_argument('-drop_out', type=float, default=0.05, help="")
    parser.add_argument('-epochs', type=int, default=50, help="训练轮次")
    parser.add_argument('-loss_type', type=str, default='')

    # 共8449个样本
    parser.add_argument('-train_batch_size', type=int, default=32, help="train批次大小")
    # 2785个
    parser.add_argument('-test_batch_size', type=int, default=16, help="test批次大小")
    parser.add_argument('-valid_batch_size', type=int, default=16, help="dev批次大小")
    parser.add_argument('-save_path', type=str, default='./save_models/')

    # model
    parser.add_argument('-input_size', type=int, default=7, help='特征个数')
    # 512
    parser.add_argument('-hidden_size', type=int, default=256, help="隐藏层单元数")
    # 3
    parser.add_argument('-laryer_num', type=int, default=3)

    # device
    parser.add_argument('-use_gpu', type=bool, default=True)
    parser.add_argument('-device', type=int, default=0, help="只设置最多支持单个gpu训练")

    # option
    parser.add_argument('-train', type=bool, default=True)
    parser.add_argument('-test', type=bool, default=True)
    parser.add_argument('-predict', type=bool, default=True)
    args = parser.parse_args()
    return args
