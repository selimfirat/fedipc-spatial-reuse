from torch import nn
import torch.nn.functional as F
from torch.nn import LayerNorm

class Transformer(nn.Module):

    def __init__(self,output_size, nhead,num_layers,dropout,device,scenario, **params):
        super(Transformer, self).__init__()
        if scenario ==1:
            self.d_model = 8
        elif scenario ==2:
            self.d_model = 14
        self.output_size = output_size
        self.encoder_layer = nn.TransformerEncoderLayer(dropout=dropout,d_model=self.d_model, nhead=nhead,batch_first=True,device=device)
        self.encoder_norm = LayerNorm(self.d_model)
        self.transfor_encode = nn.TransformerEncoder(self.encoder_layer, num_layers,norm=self.encoder_norm)
        self.linear = nn.Linear(self.d_model, self.output_size)

    def forward(self,x):
        x= x.unsqueeze(0)
        x = self.transfor_encode(x)
        x =x.squeeze(0)
        x = self.linear(F.relu(x))
        return x

