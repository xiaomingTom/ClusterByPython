#coding:utf-8
import numpy as ny
import scipy.io as sio
import warnings
from Evaluate import evaluate
from MSPL import MSPL
from Normalize import Normalize

matFile=sio.loadmat("D:\dataSet\handwritten.mat")
dataSet=[]
dataSet.append(ny.mat(matFile['mor']).T)
dataSet.append(ny.mat(matFile['fourier']).T)
dataSet.append(ny.mat(matFile['pixel']).T)
dataSet.append(ny.mat(matFile['kar']).T)
dataSet.append(ny.mat(matFile['profile']).T)
dataSet.append(ny.mat(matFile['zer']).T)
nor=Normalize()
dataSet=map(lambda x:x*15,nor.normalize(dataSet))
#dataSet=[dataSet[i]*20 for i in range(len(dataSet))]
#dataSet=map(lambda x:x*10,nor.normalize2(dataSet))
'''
dims=[dataSet[i].shape[0] for i in range(len(dataSet))]
centroids=[ny.mat(ny.zeros((dims[i],10))) for i in range(len(dataSet))]
for i in range(10):
    index=int(ny.random.rand()*200)+i*200
    #index=int(ny.random.rand()*2000)
    for j in range(len(dataSet)):
        centroids[j][:,i]=dataSet[j][:,index]
'''
gnd=matFile['gnd']
realAssment=[]
temp=ny.eye(10)
for i in range(gnd.shape[0]):
    realAssment.append(temp[gnd[i,0]].tolist())#创建真实分配矩阵
print '真实分配矩阵创建完毕，开始聚类'
mspl=MSPL(10,1.2,3,dataSet)
pur=[]
acc=[]
nmi=[]
warnings.filterwarnings('error')
for i in range(20):
    try:
        mspl.Mspl(1.2)
        p,a,n=evaluate(mspl.Assment, ny.mat(realAssment).T)
        pur.append(p)
        acc.append(a)
        nmi.append(n)
    except:
        print '有警告!'
print pur 
print acc
print nmi
print 'pur mean(std) max min std',('%.3f(%.3f)'%(ny.mean(pur),ny.std(pur))),max(pur),min(pur)
print 'acc mean(std) max min std',('%.3f(%.3f)'%(ny.mean(acc),ny.std(acc))),max(acc),min(acc)
print 'nmi mean(std) max min std',('%.3f(%.3f)'%(ny.mean(nmi),ny.std(nmi))),max(nmi),min(nmi)
input("回车结束程序!")