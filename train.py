import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from environment import enviroment
from npiCore import npi
#npiParam:[stateDim,programDim,hiddenDim,keyDim,argumentDim,lstmLayers]
#data:[ nBucket, a, trace,state ]
nProgram = 4
class taskEncoder(nn.Module):
    def __init__(self):
        pass
def eval():
    pass
def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #npiParam = [12,4,16,4,1,2]
    npiModel = npi(12,4,128,4,2,2)
    npiModel.to(device)
    lossFunction = nn.NLLLoss()
    optimizer = optim.SGD(npiModel.parameters(),lr=0.001)
    #data
    infile = open("data.pik",'rb')
    data = pickle.load(infile)
    size = 0
    for parameter in npiModel.parameters():
        size+=len(parameter)
    print("dasd")
    print(size)
    infile.close()
    #test
    #with torch.no_grad():
        #_,_,trace,state = data[0]
        #progEmb = F.one_hot(torch.tensor(trace[0][0]),4)#simple embedding using one hot
        #print(trace[0][1])
        #argsEmb = F.one_hot(torch.tensor(trace[0][1]),4)#simple embedding using one hot
        #hidden = (torch.randn(2, 1, 256),torch.randn(2, 1, 256))
        #print("ok")
        #print(state[0])
        #print("ok")
        #hidden,ter,key,arg = npiModel(torch.LongTensor(state[0]).unsqueeze(0),progEmb.unsqueeze(0), hidden)
        #print([hidden,ter,key,arg])
    
    for epoch in range(3000):
        print("start a epoch")
        #totalTrace = 0
        #progIDErr = 0
        argLossEp = 0.0
        progLossEp = 0.0
        terLossEp = 0.0
        step=0
        #print(len(data))
        if (epoch+1)%300==0:
            torch.save(npiModel.state_dict(), "model"+str(epoch))
        for example in data:
            step+=1
            print(step)
            _,_,trace,state = example
            prog=[]
            for i in range(len(trace)):
                prog.append(trace[i][0])
            #print(prog)
            npiModel.zero_grad()
            optimizer.zero_grad()
            argLoss = 0.0
            progLoss = 0.0
            terLoss = 0.0
            totalLoss = 0.0
            state = torch.LongTensor(state).unsqueeze(1).to(device)
            #print(state.shape)
            hidden = (torch.randn(2, 1, 128).to(device),torch.randn(2, 1, 128).to(device))
            progEmb = F.one_hot(torch.LongTensor(prog),4).unsqueeze(1).to(device)#simple embedding using one hot
            for i in range(len(trace)-1):
                #print(i)
                #progEmb = F.one_hot(torch.tensor(trace[i][0]),4)#simple embedding using one hot
                #hidden = (torch.randn(2, 1, 256),torch.randn(2, 1, 256))
                #hidden,ter,key,arg = npiModel(torch.LongTensor(state[i]).unsqueeze(0),progEmb.unsqueeze(0), hidden)
                #print(state[i])
                #print(progEmb[i])
                hidden,ter,key,arg = npiModel(state[i],progEmb[i], hidden)
                #print(ter)
                terGroundtruth = 1 if trace[i+1][2] else 0
                terLoss=nn.CrossEntropyLoss()(ter,torch.tensor([terGroundtruth]).to(device))
                terLossEp+=terLoss
                #print("terLoss:")
                #print(terLoss)
                progLoss=nn.CrossEntropyLoss()(key,torch.tensor([trace[i+1][0]]).to(device))
                progLossEp+=progLoss
                #print("progLoss:")
                #print(progLoss)
                if trace[i+1][0]==3:
                    #print(torch.squeeze(arg))
                    argsEmb = F.one_hot(torch.tensor(trace[i+1][1][0]),2).to(device)#simple embedding using one hot
                    
    
                    argLoss=nn.MSELoss()(torch.squeeze(arg),argsEmb.float())
                    argLossEp+=argLoss
                    #print("argLoss:")
                    #print(argLoss)
                else:
                    pass
                totalLoss = terLoss+progLoss+argLoss
                #print(totalLoss)
                totalLoss.backward(retain_graph=True)
                optimizer.step()
        print([
                argLossEp,
                progLossEp,
                terLossEp,
            ])

if __name__ == "__main__":
    train()

    