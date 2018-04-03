#coding:utf-8
import numpy as ny
import scipy.io as sio
from Evaluate import evaluate
from MSPL import MSPL

matFile=sio.loadmat("D:\dataSet\handwritten.mat")
dataSet=[]
dataSet.append(ny.mat(matFile['fourier']).T)
dataSet.append(ny.mat(matFile['kar']).T)
dataSet.append(ny.mat(matFile['pixel']).T)
dataSet.append(ny.mat(matFile['profile']).T)
gnd=matFile['gnd']
realAssment=[]
temp=ny.eye(10)
for i in range(gnd.shape[0]):
    realAssment.append(temp[gnd[i,0]].tolist())#创建真实分配矩阵
mspl=MSPL(10,0.007,1.5,dataSet)
print '真实分配矩阵创建完毕，开始聚类'
mspl.mspl()
evaluate(mspl.Assment, ny.mat(realAssment).T)