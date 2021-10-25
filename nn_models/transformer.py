from torch import nn
import torch.nn.functional as F
from torch.nn import LayerNorm

class Transformer(nn.Module):

    def __init__(self,output_size, nhead,num_layers,dropout,device,d_model=8, **params):
        super(Transformer, self).__init__()
        self.output_size = output_size
        self.encoder_layer = nn.TransformerEncoderLayer(dropout=dropout,d_model=d_model, nhead=nhead,batch_first=True,device=device)
        encoder_norm = LayerNorm(d_model)
        self.transfor_encode = nn.TransformerEncoder(self.encoder_layer, num_layers,norm=encoder_norm)
        self.linear = nn.Linear(d_model, self.output_size)

    def forward(self,x):
        x= x.unsqueeze(0)
        x = self.transfor_encode(x)
        x =x.squeeze(0)
        x = self.linear(F.relu(x))
        return x

