#coding:utf-8
import scipy.io as sio
from SPL_kmeans import kMeans2
import numpy as ny
from Evaluate import evaluate
from Normalize import Normalize
from MSPL import MSPL

matFile=sio.loadmat("D:\dataSet\yale_mtv.mat")
dataSet=[ny.mat(matFile['X'][0,0]),ny.mat(matFile["X"][0,1]),ny.mat(matFile["X"][0,2])]
nor=Normalize()
dataSet=nor.normalize(dataSet)
realAssment=ny.mat(ny.zeros((15,165)))
gt=matFile['gt']
etmp=ny.mat(ny.eye(15))
for i in range(gt.shape[0]):
    realAssment[:,i]=etmp[:,gt[i,0]-1]
dims=[dataSet[i].shape[0] for i in range(len(dataSet))]
centroids=[ny.mat(ny.zeros((dims[i],15))) for i in range(len(dataSet))]
for i in range(15):
    index=int(ny.random.rand()*165)
    for v in range(len(dataSet)):
        centroids[v][:,i]=dataSet[v][:,index]
dataSet2=dataSet[0]
centroids2=centroids[0]
for i in range(1,len(dataSet)):
    dataSet2=ny.vstack((dataSet2,dataSet[i]))
    centroids2=ny.vstack((centroids2,centroids[i]))
for i in range(len(dataSet)):
    cenTmp,assment=kMeans2(dataSet[i], 15,centroids[i])
    print 'kmeans view',i
    evaluate(assment, realAssment)
print '\n Con-Mc'
cenTmp,assment=kMeans2(dataSet2, 15,centroids2)
evaluate(assment, realAssment)
dataSet=map(lambda x:x*5,dataSet)
centroids=map(lambda x:x*5,centroids)
print '\n mspl'
mspl=MSPL(15,0.05,1.3,dataSet,centroids)
mspl.mspl2()
evaluate(mspl.Assment, realAssment)