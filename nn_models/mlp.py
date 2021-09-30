import torch


class MLP(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, **params):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)

        return output

    @staticmethod
    def init_model(**params):

        return MLP(**params)
