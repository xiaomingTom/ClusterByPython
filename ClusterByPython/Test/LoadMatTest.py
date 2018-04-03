#coding:utf-8
#sys.stdout=codecs.getwriter('utf8')(sys.stdout)
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
dataMat=load_data['fourier']
'''
dataMat=ny.hstack((dataMat,load_data['kar']))
dataMat=ny.hstack((dataMat,load_data['mor']))
dataMat=ny.hstack((dataMat,load_data['pixel']))    
dataMat=ny.hstack((dataMat,load_data['profile']))
dataMat=ny.hstack((dataMat,load_data['zer'])) 
'''
print '数据加载完毕'               
gnd=load_data['gnd']
realAssment=[]
temp=ny.eye(10)
#dataMat=cut(dataMat)
#gnd=cut(gnd)
for i in range(gnd.shape[0]):
    realAssment.append(temp[gnd[i,0]].tolist())#创建真实分配矩阵
#print dataMat.shape
print '真实分配矩阵创建完毕，开始聚类'
centroids,Assment=kMeans(ny.mat(dataMat).T, 10)
print '聚类结束'
evaluate(Assment, ny.mat(realAssment).T)
