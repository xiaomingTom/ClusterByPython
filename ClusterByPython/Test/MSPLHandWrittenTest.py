#coding:utf-8
import numpy as ny
import scipy.io as sio
from Evaluate import evaluate
from MSPL import MSPL
from Normalize import Normalize

matFile=sio.loadmat("D:\dataSet\handwritten.mat")
dataSet=[]
#dataSet.append(ny.mat(matFile['mor']).T)
#dataSet.append(ny.mat(matFile['fourier']).T)
dataSet.append(ny.mat(matFile['pixel']).T)
#dataSet.append(ny.mat(matFile['kar']).T)
#dataSet.append(ny.mat(matFile['profile']).T)
#dataSet.append(ny.mat(matFile['zer']).T)
nor=Normalize()
dataSet=nor.normalize(dataSet)
dataSet=[dataSet[i]*20 for i in range(len(dataSet))]
dims=[dataSet[i].shape[0] for i in range(len(dataSet))]
centroids=[ny.mat(ny.zeros((dims[i],10))) for i in range(len(dataSet))]
for i in range(10):
    index=int(ny.random.rand()*200)+i*200
    for j in range(len(dataSet)):
        centroids[j][:,i]=dataSet[j][:,index]
gnd=matFile['gnd']
realAssment=[]
temp=ny.eye(10)
for i in range(gnd.shape[0]):
    realAssment.append(temp[gnd[i,0]].tolist())#创建真实分配矩阵
print '真实分配矩阵创建完毕，开始聚类'
for i in range(len(dataSet)):
    print '\n'
    mspl=MSPL(10,0.05,1.3,dataSet[i:i+1],centroids[i:i+1])
    mspl.mspl()
    print 'view',i
    evaluate(mspl.Assment, ny.mat(realAssment).T)
mspl=MSPL(10,0.05,1.3,dataSet,centroids)
mspl.mspl()
evaluate(mspl.Assment, ny.mat(realAssment).T)
'''
dataSet[0]=dataSet[0]*10
centroids[0]=centroids[0]*10
dataSet[1]=dataSet[1]*0.7
centroids[1]=centroids[1]*0.7
dataSet[3]=dataSet[3]*0.1/3
centroids[3]=centroids[3]*0.1/3
mspl=MSPL(10,0.05,1.4,dataSet,centroids)
mspl.mspl()
evaluate(mspl.Assment, ny.mat(realAssment).T)
'''