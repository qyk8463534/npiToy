import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class npi(nn.Module):
    def __init__(self,stateDim,programDim,hiddenDim,keyDim,argumentDim,lstmLayers):
        super(npi, self).__init__()
        self.stateDim = stateDim
        self.programDim = programDim
        self.hiddenDim = hiddenDim
        self.keyDim = keyDim
        self.argumentDim = argumentDim
        self.inputDim = stateDim+programDim
        self.lstmLayers = lstmLayers
        #layers
        self.lstm = nn.LSTM(input_size=self.inputDim,hidden_size=self.hiddenDim,num_layers=self.lstmLayers)
        self.ter = nn.Sequential(
            nn.Linear(self.hiddenDim,self.hiddenDim),
            #nn.ReLU(),
            nn.Linear(self.hiddenDim,2),
            nn.ReLU()
        )
        self.key = nn.Sequential(
            nn.Linear(self.hiddenDim,self.hiddenDim),
            nn.ReLU(),
            nn.Linear(self.hiddenDim,self.keyDim),
            nn.ReLU()
            )
        self.arg = nn.Sequential(
            nn.Linear(self.hiddenDim,self.hiddenDim),
            #nn.ReLU(),
            nn.Linear(self.hiddenDim,self.argumentDim),
            nn.ReLU()
        )

    def forward(self, stateEncoding,programEmbedding,hidden):
        spIn = torch.cat([stateEncoding,programEmbedding], -1)
        #print(stateEncoding)
        #print(spIn.shape)
        #print(hidden)
        #print(spIn.unsqueeze(1).shape)
        out, hidden = self.lstm(spIn.unsqueeze(1).float(),hidden)
        embedding = F.relu(out)
        ter = self.ter(embedding).squeeze(1).squeeze(1)
        key = self.key(embedding).squeeze(1)
        arg = self.arg(embedding).squeeze(1)
        return hidden,ter,key,arg
        

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
if __name__ == '__main__':
    state = torch.randn(3, 4)
    prog = torch.randn(3, 6)
    core = npi(stateDim=4,
                   programDim=6,
                   hiddenDim=5,
                   lstmLayers=2,
                   keyDim=4,
                   argumentDim=5)
    hidden = torch.zeros(2, 1, 5),torch.zeros(2, 1, 5)
    h,ret, pkey, args = core(state, prog, hidden)
    print(h)
    print(ret)
    print(pkey)
    print(args)