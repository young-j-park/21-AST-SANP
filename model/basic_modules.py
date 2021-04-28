
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_units, output_size):
        super(MLP, self).__init__()
        self.L = len(hidden_units)
        if self.L == 0:
            layers = [Dense(input_size, output_size)]
        else:
            layers = [Dense(input_size, hidden_units[0]), nn.ReLU()]
            for i in range(self.L - 1):
                layers.append(Dense(hidden_units[i], hidden_units[i + 1]))
                layers.append(nn.ReLU())
            layers.append(Dense(hidden_units[-1], output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Dense(nn.Linear):
    def __init__(self, inp_size, otp_size):
        super().__init__(inp_size, otp_size)
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)


class LinEmbedding(nn.Linear):
    def __init__(self, inp_size, otp_size):
        super().__init__(inp_size, otp_size, bias=False)
        torch.nn.init.xavier_uniform_(self.weight)
