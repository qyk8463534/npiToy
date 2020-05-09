import pickle
import numpy as np
from task import Trace
from environment import enviroment
def generate(nExamples,arrayLength,nBucket):
    data=[]
    for i in range(nExamples):
        a = np.zeros(arrayLength)
        for j in range(arrayLength):
            a[j]=np.random.rand()
        #print(a)
        trace = Trace(nBucket,a).trace
        state = enviroment(nBucket,a,trace).historystate
        data.append(( nBucket, a, trace,state ))

    with open('./{}.pik'.format("data"), 'wb') as f:
        pickle.dump(data, f)
        f.close()
if __name__ == '__main__':
    x=[0.3,0.4,0.1,0.7,0.5,0.8,0.6,0.2]
    generate(10,10,10)
    #infile = open("data.pik",'rb')
    #new = pickle.load(infile)
    #infile.close()
    #print(new[1])