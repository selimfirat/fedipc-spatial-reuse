from torch import nn
import torch.nn.functional as F

class Transformer(nn.Module):

    def __init__(self,output_size, nhead,num_layers,device,d_model=8, **params):
        super(Transformer, self).__init__()
        self.output_size = output_size
        self.encoder_layer = nn.TransformerEncoderLayer(dropout=0.5,d_model=d_model, nhead=nhead,batch_first=True,device=device)
        self.transfor_encode = nn.TransformerEncoder(self.encoder_layer, num_layers)

        self.linear = nn.Linear(d_model, self.output_size)
    def forward(self,x):
        x= x.unsqueeze(0)
        x = self.transfor_encode(x)
        x =x.squeeze(0)
        x = self.linear(F.relu(x))
        return x

