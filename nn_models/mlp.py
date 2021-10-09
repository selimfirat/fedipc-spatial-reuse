from torch import nn

class MLP(nn.Module):

    def __init__(self, output_size, mlp_hidden_sizes, mlp_activation, input_size, **params):
        super(MLP, self).__init__()
        self.linear_sizes = [input_size] + mlp_hidden_sizes + [output_size]

        self.fcs = nn.ModuleList([nn.Linear(self.linear_sizes[i], self.linear_sizes[i+1]) for i in range(len(self.linear_sizes) - 1) ])

        self.activation = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU
        }[mlp_activation]()

    def forward(self, x):
        h = x

        for fc in self.fcs:
            h = fc(h)
            h = self.activation(h)

        return h
