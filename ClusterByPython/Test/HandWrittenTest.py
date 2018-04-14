#coding:utf-8
import numpy as ny
import scipy.io as sio
from Evaluate import evaluate
from MSPL import MSPL
from Normalize import Normalize
from SPL_kmeans import kMeans

matFile=sio.loadmat("D:\dataSet\handwritten.mat")
dataSet=[]
dataSet.append(ny.mat(matFile['mor']).T)
dataSet.append(ny.mat(matFile['fourier']).T)
dataSet.append(ny.mat(matFile['pixel']).T)
dataSet.append(ny.mat(matFile['kar']).T)
dataSet.append(ny.mat(matFile['profile']).T)
dataSet.append(ny.mat(matFile['zer']).T)
nor=Normalize()
dataSet=map(lambda x:x*20,nor.normalize(dataSet))
dims=[dataSet[i].shape[0] for i in range(len(dataSet))]
centroids=[ny.mat(ny.zeros((dims[i],10))) for i in range(len(dataSet))]
for i in range(10):
    index=int(ny.random.rand()*200)+i*200
    for j in range(len(dataSet)):
        centroids[j][:,i]=dataSet[j][:,index]
dataSet2=dataSet[0]
centroids2=centroids[0]
for i in range(1,len(dataSet)):
    dataSet2=ny.vstack((dataSet2,dataSet[i]))
    centroids2=ny.vstack((centroids2,centroids[i]))
gnd=matFile['gnd']
realAssment=[]
temp=ny.eye(10)
for i in range(gnd.shape[0]):
    realAssment.append(temp[gnd[i,0]].tolist())#创建真实分配矩阵
print '真实分配矩阵创建完毕，开始聚类'
for i in range(len(dataSet)):
    print '\n'
    cenTmp,assment=kMeans(dataSet[i], 10,centroids[i])
    print 'kmeans view',i
    evaluate(assment, ny.mat(realAssment).T)
    #print cenTmp
    #print centroids[i]
    print '\n'
    mspl=MSPL(10,0.05,1.5,dataSet[i:i+1],centroids[i:i+1])
    mspl.mspl()
    print 'mspl view',i
    evaluate(mspl.Assment, ny.mat(realAssment).T)
cenTmp,assment=kMeans(dataSet2, 10,centroids2)
print '\n Con-Mc'
evaluate(assment, ny.mat(realAssment).T)
mspl=MSPL(10,0.05,1.5,dataSet,centroids)
mspl.mspl()
print '\n mspl'
evaluate(mspl.Assment, ny.mat(realAssment).T)
