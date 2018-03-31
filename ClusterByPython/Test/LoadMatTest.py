#coding=utf-8
import scipy.io as sio
import numpy as ny
from SPL_kmeans import kMeans
from Evaluate import evaluate

def cut(data=ny.array([[]])):
    dataCut=data[0:50]
    for i in range(1,10):
        dataCut=ny.vstack((dataCut,data[i*200:i*200+50]))
    return dataCut

load_data=sio.loadmat("D:\dataSet\handwritten.mat")
dataMat=load_data['pixel']
gnd=load_data['gnd']
realAssment=[]
temp=ny.eye(10)
#dataMat=cut(dataMat)
#gnd=cut(gnd)
for i in range(gnd.shape[0]):
    realAssment.append(temp[gnd[i,0]].tolist())#创建真实分配矩阵
#print dataMat.shape
centroids,Assment=kMeans(ny.mat(dataMat).T, 10)
print '聚类结束'
evaluate(Assment, ny.mat(realAssment).T, ny.mat(dataMat).T, centroids)