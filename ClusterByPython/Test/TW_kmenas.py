#coding:utf-8
import numpy as ny
import scipy.io as sio
import warnings
#import time
from Normalize import Normalize
from Evaluate import evaluate

class TW_kmeans:
    def __init__(self,centerNum,Lambda,Eta,dataSet=[ny.array([[]])]):
        self.setPara(centerNum,Lambda,Eta,dataSet)
        
    def setPara(self,centerNum,Lambda,Eta,dataSet=[ny.array([[]])]):
        self.centerNum=centerNum
        self.dataSet=dataSet
        self.viewNum=len(dataSet)
        self.dataNum=dataSet[0].shape[1]
        self.dims=map(lambda x:x.shape[0],dataSet)
        self.Lambda=Lambda
        self.Eta=Eta
        self.W=ny.array([1./self.viewNum]*self.viewNum)
        self.V=map(lambda x:ny.array([1./x]*x),self.dims)
        self.assment=ny.zeros((centerNum,self.dataNum))
        
    def createCentroids(self):
        self.centroids=map(lambda x:ny.zeros((x,self.centerNum)),self.dims)
        for i in range(self.centerNum):
            index=int(ny.random.rand()*self.dataNum)
            for v in range(self.viewNum):
                self.centroids[v][:,i]=self.dataSet[v][:,index]
    
    def setCen(self,dataIndex,cenIndex):
        for i in range(self.viewNum):
            self.centroids[i][:,cenIndex]=self.dataSet[i][:,dataIndex]
    
    def Cent(self):
        self.centroids=map(lambda x:ny.zeros((x,self.centerNum)),self.dims)
        index=int(ny.random.rand()*self.dataNum)
        self.setCen(index,0)
        minDist=[ny.inf]*self.dataNum
        dist=ny.array([0.]*self.dataNum)
        for i in range(1,self.centerNum):
            dist*=0
            #求各点到新聚类中心的距离
            for v in range(self.viewNum):
                dist+=((self.dataSet[v].T-self.centroids[v][:,i-1])**2).sum(1)
            minDist=map(lambda x,y:min(x,y),minDist,dist)
            probList=ny.array(minDist)/sum(minDist)
            index=ny.random.multinomial(1,probList).tolist().index(1)
            self.setCen(index, i)
        
    def updateAssment(self):
        changeFlag=False;
        for i in range(self.dataNum):
            dist=ny.array([0.]*self.centerNum)
            for v in range(self.viewNum):
                dist+=((self.centroids[v].T-self.dataSet[v][:,i])**2*self.V[v]*self.W[v]).sum(1)
            index=ny.argmin(dist)
            if self.assment[index,i]!=1:
                changeFlag=True
                self.assment[:,i]=0
                self.assment[index,i]=1
        return changeFlag
    
    def updateCentroids(self):
        for v in range(self.viewNum):
            self.centroids[v]=self.dataSet[v].dot(self.assment.T)/(self.assment.T.sum(0))

    def updateV(self):
        for v in range(self.viewNum):
            E=ny.e**(-((self.dataSet[v]-self.centroids[v].dot(self.assment))**2*self.W[v]).sum(1)/self.Eta)
            self.V[v]=E/E.sum()
    
    def updateW(self):
        for v in range(self.viewNum):
            self.W[v]=ny.e**(-((self.dataSet[v]-self.centroids[v].dot(self.assment)).T**2*self.V[v]).sum()/self.Lambda)
        self.W/=self.W.sum()
    
    def kmeans(self):
        times=0
        self.V=map(lambda x:x/x[0],self.V)
        #self.createCentroids()
        self.Cent()
        while self.updateAssment():
            self.updateCentroids()
            times+=1
        print times

    def tw_kmeans(self):
        times=0
        self.createCentroids()
        #self.Cent()
        while self.updateAssment():
            self.updateCentroids()
            self.updateV()
            self.updateW()
            times+=1
        print times
        
warnings.filterwarnings('error')  
matFile=sio.loadmat("D:\dataSet\handwritten.mat")
dataSet=[]
dataSet.append(matFile['mor'].T)
dataSet.append(matFile['fourier'].T)
dataSet.append(matFile['pixel'].T)
dataSet.append(matFile['kar'].T)
dataSet.append(matFile['profile'].T)
dataSet.append(matFile['zer'].T)
nor=Normalize()
dataSet=map(ny.array,nor.normalize(dataSet))
gnd=matFile['gnd']
realAssment=[]
temp=ny.eye(10)
for i in range(gnd.shape[0]):
    realAssment.append(temp[gnd[i,0]].tolist())
tw=TW_kmeans(10,30,7,dataSet)
pur=[]
acc=[]
nmi=[]
for i in range(10):
    try:
        tw.kmeans()
        #tw.tw_kmeans()
        p,a,n=evaluate(ny.mat(tw.assment), ny.mat(realAssment).T)
        pur.append(p)
        acc.append(a)
        nmi.append(n)
    except:
        print '有警告'
        continue
print pur 
print acc
print nmi
print 'pur mean(std) max min std',('%.3f(%.3f)'%(ny.mean(pur),ny.std(pur))),max(pur),min(pur)
print 'acc mean(std) max min std',('%.3f(%.3f)'%(ny.mean(acc),ny.std(acc))),max(acc),min(acc)
print 'nmi mean(std) max min std',('%.3f(%.3f)'%(ny.mean(nmi),ny.std(nmi))),max(nmi),min(nmi)