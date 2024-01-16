import os

import numpy as np
import torch

from hybrid.utils.check import check_shape
from hybrid.utils.metrics import metric


def load_ckpt(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    return model, checkpoint['epoch'], checkpoint['dev_loss']


def predict_step(args, model, batch_X, batch_y, scaler, sample_plot):
    batch_X, batch_y = batch_X.to(args.device), batch_y.to(args.device)
    pred_y = model(batch_X)
    batch_y, pred_y = scaler.inverse_transform(batch_y), scaler.inverse_transform(pred_y)
    loss = model.loss_fn(pred_y, batch_y)
    batch_X = scaler.inverse_transform(batch_X)

    if len(sample_plot) == 0:  # (b,nx,o) (b,ny,o)
        sample_plot = [batch_X.cpu().numpy(), batch_y.cpu().numpy(), pred_y.cpu().numpy()]
    return pred_y, batch_y, sample_plot, loss


def predict(args, model, test_loader, scaler):
    device = args.device
    PATH = args.ckpt_path + args.gru_path
    model.load_state_dict(torch.load(PATH))
    # model, best_epoch, dev_loss_min = load_ckpt(PATH, model)
    model.to(device)
    model.eval()

    preds, trues = [], []
    sample_plot = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            pred_y, batch_y, sample_plot, loss = predict_step(args, model, batch_X, batch_y, scaler, sample_plot)
            preds.append(pred_y)
            trues.append(batch_y)

        res_path = args.res_path  # + setting + '/'
        if not os.path.exists(res_path):
            os.makedirs(res_path)  # creates all the intermediate directories

        preds = torch.cat(preds, dim=0).cpu().numpy()
        trues = torch.cat(trues, dim=0).cpu().numpy()
        mae, mse, rmse = metric(preds, trues)
        print(f'test mse:{mse:.3f}, mae:{mae:.3f}')
        np.save(res_path + 'metrics.npy', np.array([mae, mse, rmse]))
        np.save(res_path + 'pred.npy', preds)
        np.save(res_path + 'true.npy', trues)

    return sample_plot, {'mse': mse, 'mae': mae}
