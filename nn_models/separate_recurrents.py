import torch
from torch import nn


class SeparateRecurrentsModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, **params):
        super(SeparateRecurrentsModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        model_ins = nn.GRU

        self.model = model_ins(768, 32, 2, batch_first=True, bidirectional=True)

        self.linear = nn.Linear(64, self.output_size)

    def forward(self, x):

        print(x)

