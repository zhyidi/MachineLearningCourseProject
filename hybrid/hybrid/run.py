import os
import random
from datetime import datetime

import numpy as np
import torch

from config import parse_args
from hybrid.data.data_loader import load_data, process_data
from hybrid import draw
from hybrid import tester
from hybrid.trainer import Trainer

from hybrid.models.rnn import Seq2SeqEncoder, Seq2SeqDecoder, Seq2Seq


def create_model(args):
    device = args.device
    input_size, output_size, hidden_size, num_layers = args.input_size, args.output_size, args.hidden_size, args.num_layers
    dropout = args.dropout
    encoder = Seq2SeqEncoder(input_size, hidden_size, output_size, num_layers, dropout)
    decoder = Seq2SeqDecoder(hidden_size, output_size, args.timestep_x, args.timestep_y, dropout)
    model = Seq2Seq(encoder, decoder).to(device)
    return model


def search(args):
    config = {
        # 'lr': random.choice([1e-3, 5e-4, 1e-4]),
        'hidden_size': random.choice([64, 128, 256]),
        'batch_size': random.choice([16, 32, 64]),
        'dropout': random.choice([0.7, 0.9]),
        # 'lr': random.choice([10 ** -(i + 1) for i in range(4)]),
        # 'num_layers': random.choice([1,2,3]),
        # "dropout": random.uniform(),
        # "l1": random.uniform(),
    }
    for k, v in config.items():
        setattr(args, k, v)
        # getattr(args, k)
    return args, config


def train_test(args, train_loader, dev_loader, test_loader, scaler):
    args.is_train = True
    args.is_test = args.is_draw = True

    model = create_model(args)
    print(model)
    if args.is_train:
        start_time = datetime.now()
        trainer = Trainer(args, model, train_loader, dev_loader, scaler)
        train_loss_list, dev_loss_list = trainer.fit()
        print(f'training time: {datetime.now() - start_time}')
        if not args.is_tune:
            draw.draw_learning_curve(args.max_epochs, train_loss_list, dev_loss_list)
    if args.is_test:
        start_time = datetime.now()
        sample_plot, test_metrics = tester.predict(args, model, test_loader, scaler)
        print(f'test time: {datetime.now() - start_time}')
        if not args.is_tune:
            draw.draw_prediction_curve(args.timestep_x, args.timestep_y, sample_plot)
    return test_metrics


def main():
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # torch.use_deterministic_algorithms(True)
    # torch.set_deterministic(True)
    # torch.manual_seed(13)
    # np.random.seed(13)

    args = parse_args()
    if torch.cuda.is_available():
        args.device = torch.device('cuda:4')
    else:
        raise Exception('cuda is not available')
    print('device:', args.device)

    train_data, dev_data, test_data = load_data(args.root_path)
    train_loader, dev_loader, test_loader, scaler = process_data(args, train_data, dev_data, test_data)

    # args.is_tune = True
    if args.is_tune:
        exps = []
        for i in range(10):
            args, config = search(args)
            print(f'\nstart exp {i}, config: {config}')
            test_metrics = train_test(args, train_loader, dev_loader, test_loader, scaler)
            print(f'\nend exp {i}, config: {config}')
            exps.append((test_metrics, config))
        for exp in exps:
            print(exp)
    else:
        avg_loss = {'mse': 0.0, 'mae': 0.0}
        num_rounds = 1
        for i in range(num_rounds):
            torch.manual_seed(i)
            np.random.seed(i)
            test_metrics = train_test(args, train_loader, dev_loader, test_loader, scaler)
            print(test_metrics)
            avg_loss['mse'] += test_metrics['mse']
            avg_loss['mae'] += test_metrics['mae']
        avg_loss['mse'] /= num_rounds
        avg_loss['mae'] /= num_rounds
        print(f'avg_loss: {avg_loss}')


if __name__ == "__main__":
    main()
