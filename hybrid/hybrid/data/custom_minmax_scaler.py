import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


# transform X to X_scaled
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min
class CustomMinMaxScaler(MinMaxScaler):
    # def __init__(self, feature_range):
    #     super().__init__(feature_range)
    # self.min = None
    # self.scale = None

    # min_ = min - X.min(axis=0) * self.scale_
    # scale_ = (max - min) / (X.max(axis=0) - X.min(axis=0))
    # def fit(self, data):
    #     super().fit(data)
    # self.min = super().min_
    # self.scale = super().scale_

    def inverse_transform(self, X):
        min = torch.from_numpy(self.min_).type_as(X).to(X.device) if torch.is_tensor(X) else self.min_
        scale = torch.from_numpy(self.scale_).type_as(X).to(X.device) if torch.is_tensor(X) else self.scale_
        return (X - min) / scale


if __name__ == '__main__':
    scaler = CustomMinMaxScaler()
    data = np.random.rand(4, 2)
    print(data)

    scaler.fit(data)
    data = scaler.transform(data)
    print(data)

    data = data.reshape(2, 2, 2)
    data = scaler.inverse_transform(data)
    print('\n', data.reshape(4, 2))

    print(scaler.data_min_, scaler.data_max_)
