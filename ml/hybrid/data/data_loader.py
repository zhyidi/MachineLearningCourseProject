import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler

from hybrid.config import parse_args
from hybrid.data.custom_minmax_scaler import CustomMinMaxScaler
from hybrid.data.standard_scaler import StandardScaler


def load_data(root_path):
    train_data = pd.read_csv(root_path + 'train_set.csv').iloc[:, 1:].to_numpy()
    dev_data = pd.read_csv(root_path + 'validation_set.csv').iloc[:, 1:].to_numpy()
    test_data = pd.read_csv(root_path + 'test_set.csv').iloc[:, 1:].to_numpy()
    return train_data, dev_data, test_data


def create_dataset(data, timestep_x, timestep_y):
    data_X, data_y = [], []
    for i in range(len(data) - timestep_x - timestep_y + 1):
        data_X.append(data[i:i + timestep_x])
        data_y.append(data[i + timestep_x:i + timestep_x + timestep_y])
    data_X = torch.from_numpy(np.array(data_X)).to(torch.float32)  # shape: (num_steps, input_size)
    data_y = torch.from_numpy(np.array(data_y)).to(torch.float32)  # shape: (num_steps, input_size)
    return data_X, data_y


def process_data(args, train_data, dev_data, test_data):
    scaler = CustomMinMaxScaler(feature_range=(-1, 1))
    # scaler = StandardScaler()
    scaler.fit(train_data)
    train_scaled = scaler.transform(train_data)
    dev_scaled = scaler.transform(dev_data)
    test_scaled = scaler.transform(test_data)
    # print(train_scaled.min(), train_scaled.max())

    timestep_x, timestep_y = args.timestep_x, args.timestep_y
    train_X, train_y = create_dataset(train_scaled, timestep_x, timestep_y)
    dev_X, dev_y = create_dataset(dev_scaled, timestep_x, timestep_y)
    test_X, test_y = create_dataset(test_scaled, timestep_x, timestep_y)
    # print(test_X.shape[0])  # test_size = 2785

    train_loader = data.DataLoader(data.TensorDataset(train_X, train_y), shuffle=True, batch_size=args.batch_size)
    dev_loader = data.DataLoader(data.TensorDataset(dev_X, dev_y), shuffle=False, batch_size=args.batch_size)
    test_loader = data.DataLoader(data.TensorDataset(test_X, test_y), shuffle=False, batch_size=args.batch_size)
    return train_loader, dev_loader, test_loader, scaler


if __name__ == "__main__":
    args = parse_args()
    train_data, dev_data, test_data = load_data('./ETT-small/')
    # print(test_data.shape, test_data[:5])  # test_raw_size = 2976
    train_loader, dev_loader, test_loader, scaler = process_data(args, train_data, dev_data, test_data)

    batch_X, batch_y = next(iter(train_loader))
    print(batch_X.shape, batch_y.shape)  # (batch_size, num_steps, input_size)
