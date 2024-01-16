import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Time Series Forecasting')
    # data
    parser.add_argument('-root_path', type=str, default='./data/ETT-small/')
    parser.add_argument('-data_path', type=str, default='ETTh1')
    parser.add_argument('-ckpt_path', type=str, default='./models/checkpoints/')
    parser.add_argument('-gru_path', type=str, default='gru_checkpoint')
    parser.add_argument('-res_path', type=str, default='./results/')
    parser.add_argument('-timestep_x', type=int, default=96, help="input time steps")
    parser.add_argument('-timestep_y', type=int, default=336, help="prediction time steps") # 336

    # model
    # parser.add_argument('-model', type=str, default='RNN', help="")
    parser.add_argument('-input_size', type=int, default=7)
    parser.add_argument('-hidden_size', type=int, default=128)
    parser.add_argument('-output_size', type=int, default=7)
    parser.add_argument('-num_layers', type=int, default=1)

    # train
    parser.add_argument('-max_epochs', type=int, default=25, help="")  # 25
    parser.add_argument('-batch_size', type=int, default=64, help="")
    parser.add_argument('-lr', type=float, default=1e-4, help="")
    parser.add_argument('-dropout', type=float, default=0.9, help="") # 0.9
    # parser.add_argument('-loss', type=str, default='mse')
    # parser.add_argument('--patience', type=int, default=3, help='early stopping patience')

    # device
    parser.add_argument('-device', type=int, default=0, help="")
    # parser.add_argument('-use_gpu', type=bool, default=True)

    # option
    parser.add_argument('-is_tune', type=bool, default=False)
    parser.add_argument('-is_train', type=bool, default=False)
    parser.add_argument('-is_test', type=bool, default=False)
    parser.add_argument('-is_draw', type=bool, default=False)

    # parser.add_argument('-inspect_fit', type=bool, default=True)
    # parser.add_argument('-lr-scheduler', type=bool, default=True)
    args = parser.parse_args()
    return args
