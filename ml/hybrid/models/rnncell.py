import torch
import torch.nn as nn
from hybrid.config import parse_args

from hybrid.utils.check import check_shape
from hybrid.models.encoder_decoder import Encoder, Decoder, EncoderDecoder


def init_seq2seq(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])


class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.dropout(Y) + X
        # return self.ln(self.dropout(Y) + X)


class RNNBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0):
        super().__init__()
        self.rnn_blk = nn.GRUCell(input_size, hidden_size)
        self.addnorm = AddNorm(hidden_size, dropout)
        self.dense = nn.Linear(input_size, hidden_size)

    def forward(self, X, h):
        outputs = []
        for t in range(X.shape[0]):
            h = self.rnn_blk(X[t, :, :], h) # B,I -> B,H
            outputs.append(h.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        # check_shape(self.dense(X), outputs.shape)
        outputs = self.addnorm(self.dense(X), outputs)
        return outputs, h


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.blks = nn.Sequential()
        self.blks.add_module("block" + str(0), RNNBlock(input_size, hidden_size, dropout))
        for i in range(num_layers - 1):
            self.blks.add_module("block" + str(i + 1), RNNBlock(hidden_size, hidden_size, dropout))

    def forward(self, X, *args):  # T,B,I -> T,B,H  L,B,H
        hidden_state = []
        for i, blk in enumerate(self.blks):
            h = None
            X, h = blk(X, h)
            hidden_state.append(h.unsqueeze(0))
        hidden_state = torch.cat(hidden_state, dim=0)
        # check_shape(hidden_state, (self.num_layers, hidden_state.shape[1], self.hidden_size))
        return X, hidden_state


class Seq2SeqEncoder(Encoder):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0):
        super().__init__()
        self.rnn = RNN(input_size, hidden_size, num_layers=num_layers, dropout=dropout)
        self.apply(init_seq2seq)

    # X shape: (batch_size, num_steps, input_size); num_steps = timestep_x
    # X shape: (num_steps, batch_size, input_size)
    # outputs shape: (num_steps, batch_size, hidden_size)
    # hidden_state shape: (num_layers, batch_size, hidden_size)
    def forward(self, X, *args):
        X = X.permute(1, 0, 2)
        outputs, hidden_state = self.rnn(X)
        return outputs, hidden_state


class Seq2SeqDecoder(Decoder):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0):
        super().__init__()
        self.rnn = RNN(input_size + hidden_size, hidden_size, num_layers=num_layers, dropout=dropout)
        self.dense = nn.Linear(hidden_size, output_size)
        self.apply(init_seq2seq)

    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs

    # X shape: (batch_size, num_steps, input_size); num_steps = timestep_y
    # X shape: (num_steps, batch_size, input_size)
    # context shape: (batch_size, hidden_size)
    # Broadcast context to (num_steps, batch_size, hidden_size)
    # outputs shape: (num_steps, batch_size, hidden_size)
    # outputs shape: (batch_size, num_steps, output_size)
    # hidden_state shape: (num_layers, batch_size, hidden_size)
    def forward(self, X, state):
        X = X.permute(1, 0, 2)
        enc_output, hidden_state = state
        context = enc_output[-1]
        context = context.repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), -1)
        outputs, hidden_state = self.rnn(X_and_context, hidden_state)
        outputs = self.dense(outputs).permute(1, 0, 2)
        return outputs, [enc_output, hidden_state]  # dec_output, dec_state


class Seq2Seq(EncoderDecoder):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    # pred_y/batch_y shape: (batch_size, timestep, output_size)
    def loss_fn(self, pred_y, batch_y):
        return nn.MSELoss()(pred_y, batch_y)
