import torch
from torch import nn


class SeparateRecurrentsModel(nn.Module):

    def __init__(self, input_size, output_size, **params):
        super(SeparateRecurrentsModel, self).__init__()
        self.num_keys = int(input_size.replace("UNKNOWN_DICT_", ""))
        self.output_size = output_size

        model_ins = nn.GRU

        self.model_combined = model_ins(3, 32, 2, batch_first=True, bidirectional=True)
        self.model_interference = model_ins(1, 32, 2, batch_first=True, bidirectional=True)

        self.linear = nn.Linear(128, self.output_size)

    def forward(self, x):

        out_combined, _ = self.model_combined(x["combined"])
        out_interference, _ = self.model_interference(x["interference"])

        pooled_combined = torch.mean(out_combined, 1)
        pooled_interference = torch.mean(out_interference, 1)

        out = torch.cat([pooled_combined, pooled_interference], -1)

        out = self.linear(out)

        return out
