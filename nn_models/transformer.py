import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from time import time
class Transformer(nn.Module):

    def __init__(self,d_model,nhead,num_layers,dropout,device, **params):
        super(Transformer, self).__init__()
        self.d_model = d_model
        #print(input_size)
        self.device = device
        self.dim_size = int(self.d_model * 4)
        self.init_linear = nn.Linear(9,self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(dropout=dropout,d_model=self.d_model, dim_feedforward=self.dim_size,
                                                        nhead=nhead,batch_first=True,device=device)
        self.encoder_norm = LayerNorm(self.d_model)
        self.transfor_encode = nn.TransformerEncoder(self.encoder_layer, num_layers,norm=self.encoder_norm)
        self.linear = nn.Linear(self.d_model, 1)

    def forward(self,x):
        try:
            seq = x["input"]
        except:
            seq = self.get_sta_sequences(x)
        x = self.init_linear(seq)
        x = self.transfor_encode(x)
        x = self.linear(F.relu(x))
        x = x.squeeze(2)
        return x

    def get_sta_sequences(self,x): ## squential_features_preprocessor
        '''
        :param x: Dict
        :return: Tensor: [bs,sta,sequence]
        '''
        bs =x['combined'].size(0)
        threshold,pre_seq = x['combined'][:,0],x['combined'][:,1:]
        num_sta = int(len(pre_seq[0]) / 2)
        index = []
        [index.extend([n,num_sta+n]) for n in range(num_sta)]
        index = torch.LongTensor(index)
        threshold = threshold.view(-1,1,1).expand(-1,num_sta,1)
        pre_seq =torch.index_select(pre_seq,1,index).view(bs,num_sta,2) #[rssi,sinr]
        ##inference and padding
        zero_pad = torch.zeros((bs,6-len(x['interference'][0]))).to(self.device)
        intr = torch.cat((x['interference'],zero_pad),dim=1)
        intr = intr.unsqueeze(1).expand(-1,num_sta,6)
        sequence = torch.cat((pre_seq,threshold,intr),dim=2) ## [rssi,sinr,threshold,iterference(array)]per sta
        return sequence

