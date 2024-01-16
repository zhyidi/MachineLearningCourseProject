# -*- coding: utf-8 -*- 
# @Time : 2023/12/7 17:00 
# @Author : lf_Liu 
# @File : run.py
import os
import heapq
import torch
import datetime
from tqdm import *
import numpy as np
import torch.nn as nn
from model_v0 import LSTM
from config import parse_args
import matplotlib.pyplot as plt
from data_process import data_processing

def inverse_data(arg, scaler,data_pre,data_y):
    mean = torch.Tensor(scaler.mean_).to(arg.device)
    std = torch.Tensor(np.sqrt(scaler.var_)).to(arg.device)
    data_pre = data_pre * std.unsqueeze(0).unsqueeze(1) + mean.unsqueeze(0).unsqueeze(1)
    data_y = data_y * std.unsqueeze(0).unsqueeze(1) + mean.unsqueeze(0).unsqueeze(1)
    return data_pre,data_y

def train(args,model,train_loader,dev_loader,time):
    print(f'lr {args.lr}')
    print(f'epoch {args.epochs}')
    device = args.device
    # loss
    if args.loss_type == 'mse':
        loss_function = nn.MSELoss()
    else:
        loss_function = nn.L1Loss()
    # 给出L2正则化
    # , weight_decay = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # 定义优化器
    # 模型训练
    model_dir = args.save_path + time
    if os.path.exists(model_dir):
        pass
    else:
        os.mkdir(model_dir)
    train_loss_reco = []
    valid_loss_reco = []
    best_dev_loss = 100
    for epoch in tqdm(range(args.epochs)):
        model.train()
        running_loss = []
        # train_bar = tqdm(train_loader)  # 形成进度条
        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(device), y_train.to(device)
            y_train_pred = model(x_train).to(device)
            train_loss = loss_function(y_train_pred, y_train)
            model.zero_grad()
            train_loss.backward()
            optimizer.step()
            running_loss.append(train_loss.item())
            # train_bar.desc = "train epoch[{}/{}]".format(epoch + 1,arg.epochs)
        torch.save(model.state_dict(),model_dir + '/lstm_model' + '_' + str(epoch + 1)
                   + '_' + str(args.epochs) + '.pt')
        # 记录每个epoch的平均损失
        avg_train = sum(running_loss) / len(running_loss)
        # tqdm.write(f"\t Epoch {epoch + 1} / {args.epochs}, Train Loss: {avg_train:.4f}")
        # print(f'\tTrain Loss: {avg_train:.6f}')
        train_loss_reco.append(avg_train)

        losss = []
        # 模型验证
        model.eval()
        with torch.no_grad():
            for x_dev, y_dev in dev_loader:
                x_dev, y_dev = x_dev.to(device), y_dev.to(device)
                y_dev_pred = model(x_dev).to(device)
                loss = loss_function(y_dev_pred,y_dev)
                # inverse_pred, inverse_label = inverse_data(arg,scaler, y_dev_pred, y_dev)
                # loss = loss_function(inverse_pred, inverse_label)
                losss.append(loss.item())
        avg_dev = sum(losss) / len(losss)
        valid_loss_reco.append(avg_dev)
        # print(f'\tValid Loss: {avg_dev:.4f}')
        if avg_dev < best_dev_loss:
            best_dev_loss = avg_dev
            torch.save(model.state_dict(),model_dir + '/lstm_model_best.pt')
    # torch.save(model.state_dict(), model_dir + '/lstm_model_last.pt')
    print('Finished Training')
    # 验证集最小loss视为模型最优 -参考
    # best_model = valid_loss_reco.index(min(valid_loss_reco))
    # print(f'best model is {best_model+1}')

    max_number = heapq.nsmallest(5, valid_loss_reco)
    max_index = map(valid_loss_reco.index, heapq.nsmallest(5, valid_loss_reco))
    top_model_index = list(set(max_index))

    plt.figure(figsize=(10, 5))
    # 绘制训练损失
    plt.plot(train_loss_reco, label='Train_loss')
    # 绘制预测数据
    plt.plot(valid_loss_reco, label='Valid_loss')
    # 添加标题和图例
    plt.title("loss_"+args.loss_type)
    plt.legend()
    plt.savefig(model_dir + '/loss.png')
    # plt.show()
    return top_model_index


def do_test(arg,scaler,test_loader,time,top_model_index):
    path = './save_models/'+time+'/'
    all_losss = []
    for i in top_model_index:
        model_path = path + 'lstm_model_' + str(i+1) + '_' + str(args.epochs) + '.pt'
        model = LSTM(arg).to(arg.device)
        model.load_state_dict(torch.load(model_path))
        # loss
        if args.loss_type == 'mse':
            loss_function = nn.MSELoss()
        else:
            loss_function = nn.L1Loss()
        model.eval()  # 评估模式
        losss = []
        test_bar = tqdm(test_loader)
        for test_x, test_label in test_bar:
            seq, label = test_x.to(device), test_label.to(device)
            pred = model(seq)
            # inverse_pred,inverse_label = inverse_data(arg, scaler, pred, label)
            # mae = loss_function(inverse_pred,inverse_label)
            los = loss_function(pred,label)
            losss.append(los.item())
        avg = sum(losss) / len(losss)
        # print(f"\tTest loss({args.loss_type}){avg:.4f}")
        all_losss.append(avg)
    print(f'Test mean{np.mean(all_losss):.4f}')
    print(f'Test std{np.std(all_losss):.4f}')


def plot_pic(args,model,scaler,test_data,time):
    path = './save_models/'+time+'/'
    model_path = path+'lstm_model_best'+'.pt'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    for index in range(0,len(test_data),args.x_timestep+args.y_timestep):
        data_x, data_label = test_data[index]
        results = []
        labels = []
        visual_x = scaler.inverse_transform(data_x)
        for i in range(len(visual_x)):
            labels.append(visual_x[i][-1])
        # 调整维度
        data_x, data_label = data_x.unsqueeze(0).to(args.device), data_label.to(args.device)
        pred = model(data_x)  # [1,96(step),7]
        pred = pred.squeeze(0)
        pred = pred.detach().cpu().numpy()
        data_label = data_label.detach().cpu().numpy()
        pred = scaler.inverse_transform(pred)
        data_label = scaler.inverse_transform(data_label)
        for i in range(len(pred)):
            results.append(pred[i][-1])
            labels.append(data_label[i][-1])

        plt.figure(figsize=(10, 5))
        # 绘制历史数据
        plt.plot(labels, label='GroundTruth')
        # 绘制预测数据
        plt.plot(range(args.x_timestep, args.x_timestep + args.y_timestep), results, label='Prediction')
        # 添加标题和图例
        plt.title("Result Example")
        plt.legend()
        plt.savefig(path + 'visual_' + str(index) + '.png')
        # plt.show()



if __name__ == "__main__":

    args = parse_args()
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    args.device = device
    print("使用设备:", device)

    # 实例化模型
    model = ''
    pre_len = [96,336]
    loss_types = ['mse','mae']
    for y_timestep in pre_len:
        args.y_timestep = y_timestep
        scaler, train_loader, dev_loader, test_loader, all_data = data_processing(args)
        train_data, valid_data, test_data = all_data
        for loss_type in loss_types:
            if loss_type == 'mse':
                args.loss_type = 'mse'
            else:
                args.loss_type = 'mae'
            now_time = datetime.datetime.now()
            time = str(now_time.year) + str(now_time.month) + str(now_time.day) + str(now_time.hour) + str(
                now_time.minute)
            print(time)
            try:
                print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                model = LSTM(args).to(device)
                print(f">>>>>>>>>>>>>>>>>>>>>>>>>初始化{args.model}模型成功<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            except:
                print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型失败<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            top_model_index = []
            if args.train:
                print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型训练<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                top_model_index = train(args, model, train_loader, dev_loader, time)
            if args.predict:
                print(f">>>>>>>>>>>>>>>>>>>>>>>>>可视化{args.y_timestep}条数据<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                plot_pic(args, model, scaler, test_data, time)
            if args.test:
                print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型测试<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                # 记录五次模型在测试集的损失 最后求std
                do_test(args, scaler, test_loader, time, top_model_index)

