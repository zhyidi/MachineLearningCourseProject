import torch
import torch.nn as nn
from hybrid.config import parse_args

from hybrid.utils.check import check_shape
from hybrid.models.encoder_decoder import Encoder, Decoder, EncoderDecoder


def init_seq2seq(module):
    """Initialize weights for sequence-to-sequence learning."""
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])


class Seq2SeqEncoder(Encoder):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout)
        self.apply(init_seq2seq)  # Applies fn recursively to every submodule as well as self

    # X shape: (batch_size, num_steps, input_size); num_steps = timestep_x
    # X shape: (num_steps, batch_size, input_size)
    # outputs shape: (num_steps, batch_size, hidden_size)
    # hidden_state shape: (num_layers, batch_size, hidden_size)
    def forward(self, X, *args):
        # embs = self.embedding(X.t().type(torch.int64))  # torch.tensor.t(): only 2-D tensor will be transposed
        X = X.permute(1, 0, 2)
        outputs, hidden_state = self.rnn(X)
        return outputs, hidden_state


class Seq2SeqDecoder(Decoder):
    def __init__(self, hidden_size, output_size, timestep_x, timestep_y, dropout):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dense3 = nn.Linear(timestep_x, timestep_y)

    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs

    def forward(self, state):
        X, _ = state
        res = X
        X = self.dense1(X)  # (Tx,B,H) -> (Tx,B,H)
        X = self.act(X)
        X = self.dropout(X)
        X += res

        X = self.dense2(X).permute(1, 2, 0)  # (Tx,B,H) -> (Tx,B,O) -> (B,O,Tx)
        X = self.dense3(X).permute(0, 2, 1)  # -> (B,O,Ty) ->(B,Ty,O)
        return X


class Seq2Seq(EncoderDecoder):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    # pred_y/batch_y shape: (batch_size, timestep, output_size)
    def loss_fn(self, pred_y, batch_y):
        return nn.MSELoss()(pred_y, batch_y)

    def forward(self, enc_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        return self.decoder(dec_state)


if __name__ == "__main__":
    args = parse_args()
    # args.timestep_y = 336
    input_size, output_size, hidden_size, num_layers = args.input_size, args.output_size, args.hidden_size, args.num_layers
    batch_size, num_steps_x, num_steps_y = args.batch_size, args.timestep_x, args.timestep_y
    X = torch.zeros((batch_size, num_steps_x, input_size))
    y = torch.zeros((batch_size, num_steps_y, input_size))

    encoder = Seq2SeqEncoder(input_size, hidden_size, output_size, num_layers)
    enc_outputs, enc_state = encoder(X)
    # check_shape(enc_outputs, (num_steps_x, batch_size, hidden_size))
    # check_shape(enc_state, (num_layers, batch_size, hidden_size))

    decoder = Seq2SeqDecoder(hidden_size, output_size, args.timestep_x, args.timestep_y)
    state = decoder.init_state(encoder(X))
    dec_outputs = decoder(state)
    # check_shape(dec_outputs, (batch_size, num_steps_y, output_size))

    model = Seq2Seq(encoder, decoder)
    outputs = model(X)
    # check_shape(outputs, dec_outputs.shape)
