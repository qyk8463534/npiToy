import torch
import numpy as np
class enviroment:
    def __init__(self,nBucket,x,trace=None):
        self.nBucket = nBucket
        self.arr = []#bucket
        self.x = x#input array
        self.result = x#sorted array
        self.inputLen = len(x)
        self.trace = trace
        self.currentIndex=0#index of array elements
        self.loopCI=0
        self.bucketIndex=0#index of buckets
        self.sortedBI=-1 
        self.subBucketIndex=0#index of bucket elements
        self.output=0
        self.historystate=[]
        self.stateM = np.zeros(12)#input,bucket,output  last bit for loop finished
        for _ in range(nBucket): 
            self.arr.append([])
        if trace !=None:
            for step in trace:
                programID = step[0]
                argument = step[1]
                self.work(programID,argument)
    def put(self, index):
        index_b = int(self.nBucket * self. x[index])
        self.arr[index_b].append(self. x[index])
    def sort(self,b):
        for i in range(1, len(self. arr[b])): 
            up = self. arr[b][i] 
            j = i - 1
            while j >=0 and self. arr[b][j] > up:  
                self. arr[b][j + 1] = self. arr[b][j] 
                j -= 1
            self. arr[b][j + 1] = up      
        return self. arr[b]
    def cat(self):
        k = 0
        for i in range(self.nBucket): 
            for j in range(len(self.arr[i])): 
                self.result[k] = self.arr[i][j] 
                k += 1
        self.output=1
    def movel(self,ptr):
        if ptr ==0:
            if self.currentIndex == self.inputLen-1:
                self.loopCI+=1
            self.currentIndex = (self.currentIndex+1)%self.inputLen
            
        if ptr ==1:
            self.sortedBI+=1
            self.bucketIndex = (self.bucketIndex+1)%len(self.arr)
    def update(self):
        if self.sortedBI>=0:
            self.stateM[self.sortedBI]=1
        self.stateM[10]=self.loopCI
        self.stateM[11]=self.output
        self.historystate.append(np.copy(self.stateM))
        #print(self.stateM)
    def work(self,programID,argument):
        if programID == 0:
            self.put(self.currentIndex)
            self.update()
        elif programID == 1:
            self.sort(self.bucketIndex)
            self.update()
        elif programID == 2:
            self.cat()
            self.update()
        elif programID == 3:
            self.movel(argument[0])
            self.update()
        else:
            raise NotImplementedError
        
    def getState(self):
        return self.stateM# shape:array[12]
        