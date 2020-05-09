#there are serval subroutines:
#0.PUT ... Bucket
#1.sort inside bucket
#2.concat bucket
#3.move index to the left
#The trace is represent as (pID:integer,args:list,terminate boolean)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import environment as env
class Trace():
    def __init__(self,nBuckets,arrayInp):
        self.array = arrayInp
        self.nBuckets = nBuckets
        self.trace = []
        self.result=self.bucketSort(self.array)
    def insertionSort(self,b): 
        for i in range(1, len(b)): 
            up = b[i] 
            j = i - 1
            while j >=0 and b[j] > up:  
                b[j + 1] = b[j] 
                j -= 1
            b[j + 1] = up      
        return b
    def bucketSort(self,x): 
        arr = [] 
        slot_num = self.nBuckets
        for i in range(slot_num): 
            arr.append([])
            #self.trace.append(1,[],False) 
        index=0  
        for j in x: 
            index_b = int(slot_num * j)
            arr[index_b].append(j)
            self.trace.append((0,[],False)) #put which number into bucket
            self.trace.append((3,[0],False))
            index+=1
      
    # Sort individual buckets  
        for i in range(slot_num): 
            arr[i] = self.insertionSort(arr[i]) 
            self.trace.append((1,[],False))#sort bucket
            self.trace.append((3,[1],False))
    # concatenate the result 
        k = 0
        for i in range(slot_num): 
            for j in range(len(arr[i])): 
                x[k] = arr[i][j] 
                k += 1       
        self.trace.append((2,[],True))#concate result
        return x
if __name__ == '__main__':
    x=[0.3,0.4,0.1,0.7,0.5,0.8,0.6,0.2]
    a = Trace(10,x)
    print(a.trace)
    e = env.enviroment(nBucket=10,x=x,trace=a.trace)
    print(e.result)
