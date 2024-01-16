# -*- coding: utf-8 -*- 
# @Time : 2023/12/7 16:12 
# @Author : lf_Liu 
# @File : model.py
import torch
import torch.nn as nn
from config import parse_args

class LSTM(nn.Module):
    def __init__(self, args, batch_first=True,bidirectional=True):
        super(LSTM, self).__init__()
        self.input_dim = args.input_size
        self.hidden_dim = args.hidden_size
        self.num_layers = args.laryer_num
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.y_timestep = args.y_timestep
        # 双向
        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.relu = nn.ReLU()
        """
        input_dim 数据的特征维度 这里就是7
        hidden_dim 隐藏状态h的特征数
        num_layers 几层
        batch_first True则输入为[batch,timestep,features]
        bidirectional 默认False 即单向
        """
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=self.batch_first,
                            bidirectional=self.bidirectional,
                            )
        # dropout = 0.2
        self.fc = nn.Linear(self.num_directions * self.hidden_dim, self.y_timestep * self.input_dim)

    def forward(self, x):
        # 对应[batch,timestep,features]格式的输入 但其不受影响
        # h0_lstm = torch.zeros(self.num_layers * self.num_directions, x.shape[0], self.hidden_dim).to(x.device)
        # c0_lstm = torch.zeros(self.num_layers * self.num_directions, x.shape[0], self.hidden_dim).to(x.device)
        out, _ = self.lstm(x)
        # 此时out的为[batch,step,hidden_dim]
        # 拿出最后一个时间步的输出 如果是单向，等于输出中的h
        out = out[:,-1,:]
        # out = self.relu(out)
        out = self.fc(out)
        out = out.reshape(out.shape[0],self.y_timestep,self.input_dim)
        return out

if __name__ == "__main__":
    x = torch.randn(10,96,7)
    # y与之相同
    args = parse_args()
    model = LSTM(args)
    print(model(x).shape)
    # 输入为[batch,timestep,features]