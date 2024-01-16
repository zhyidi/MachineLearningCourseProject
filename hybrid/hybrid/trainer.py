import math
import os

import torch


from hybrid.utils.metrics import metric


class Trainer:
    def __init__(self, args, model, train_loader, dev_loader, scaler):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.scaler = scaler

    def training_step(self, batch_X, batch_y):
        batch_X, batch_y = batch_X.to(self.args.device), batch_y.to(self.args.device)
        pred_y = self.model(batch_X)
        batch_y, pred_y = self.scaler.inverse_transform(batch_y), self.scaler.inverse_transform(pred_y)
        loss = self.model.loss_fn(pred_y, batch_y)
        return pred_y, batch_y, loss

    def fit(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        train_loss_list, dev_loss_list = [], []
        dev_loss_min = math.inf
        max_epochs = self.args.max_epochs

        for epoch in range(max_epochs):
            self.model.train()
            preds, trues = [], []
            for batch_X, batch_y in self.train_loader:
                pred_y, batch_y, loss = self.training_step(batch_X, batch_y)
                preds.append(pred_y)
                trues.append(batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            preds = torch.cat(preds, dim=0).detach().cpu().numpy()
            trues = torch.cat(trues, dim=0).detach().cpu().numpy()
            mae, mse, rmse = metric(preds, trues)
            train_loss_list.append(mse)

            self.model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for batch_X, batch_y in self.dev_loader:
                    pred_y, batch_y, loss = self.training_step(batch_X, batch_y)
                    preds.append(pred_y)
                    trues.append(batch_y)
            preds = torch.cat(preds, dim=0).cpu().numpy()
            trues = torch.cat(trues, dim=0).cpu().numpy()
            mae, mse, rmse = metric(preds, trues)
            dev_loss_list.append(mse)

            if epoch % 4 == 0 or epoch == self.args.max_epochs - 1:
                print(
                    f'epoch [{epoch + 1}/{max_epochs}]-train_loss: {train_loss_list[-1]:.3f}, dev_loss: {dev_loss_list[-1]:.3f}')
            if not os.path.exists(self.args.ckpt_path):
                os.mkdir(self.args.ckpt_path)
            PATH = self.args.ckpt_path + self.args.gru_path
            if epoch >= 5 and mse < dev_loss_min:
                # checkpoint = {
                #     'epoch': epoch + 1,
                #     'dev_loss': mse,
                #     'model': self.model.state_dict(),
                #     'optimizer': optimizer.state_dict(),
                # }
                # torch.save(checkpoint, PATH + f'_{epoch + 1}_{mse:.2f}')
                torch.save(self.model.state_dict(), PATH)
                dev_loss_min = mse

        return train_loss_list, dev_loss_list
