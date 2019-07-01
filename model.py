import torch.nn as nn

_INPUT_SIZE = 3
_TIME_STEP = 34


class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(
            input_size=3,
            hidden_size=50,
            num_layers=1,
            batch_first=True
        )
        self.l1 = nn.Linear(50, 1)

    def forward(self, x):
        r_out, _ = self.rnn(x, None)
        out = self.l1(r_out[:, -1, :])
        return out
