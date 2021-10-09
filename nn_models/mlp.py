import torch


class MLP(torch.nn.Module):

    def __init__(self, output_size, **params):
        super(MLP, self).__init__()
        self.input_size = 4
        self.output_size = output_size
        self.hidden_size = 5
        self.fc1 = torch.nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        output = self.fc1(x)

        return output
