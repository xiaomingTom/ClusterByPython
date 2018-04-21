#coding:utf-8
import numpy as ny
import scipy.io as sio
from SPL_kmeans import kMeans2
from Evaluate import evaluate
from MSPL import MSPL
from Normalize import Normalize

matFile=sio.loadmat("D:\dataSet\segment_uni.mat")
dataSet=ny.mat(matFile['X']).T
dataSet2=[]
dataSet2.append(ny.mat(matFile['X'][:,0:9]).T)
dataSet2.append(ny.mat(matFile['X'][:,9:19]).T)
#nor=Normalize()
#dataSet2=map(lambda x:x*3.5,nor.normalize2(dataSet2))
dataSet2[1]*=2
Y=matFile['Y']
realAssment=[]
etmp=ny.eye(7);
for i in range(Y.shape[0]):
    realAssment.append(etmp[Y[i,0]-1])
realAssment=ny.mat(realAssment).T
centroids=ny.mat(ny.zeros((19,7)))
dims=[dataSet2[i].shape[0] for i in range(len(dataSet2))]
centroids2=[ny.mat(ny.zeros((dims[i],7))) for i in range(len(dataSet2))]
for i in range(centroids.shape[1]):
    index=int(ny.random.rand()*2310)
    centroids[:,i]=dataSet[:,index]
    for v in range(len(dataSet2)):
        centroids2[v][:,i]=dataSet2[v][:,index]
cenTmp,assment=kMeans2(dataSet, 7,centroids)
print '\n Con-Mc'
evaluate(assment, realAssment)
print
mspl=MSPL(7,0.05,1.4,dataSet2[0:1],centroids2[0:1])
print '\n view1'
mspl.mspl()
evaluate(mspl.Assment, realAssment)
print
mspl=MSPL(7,0.05,1.4,dataSet2[1:2],centroids2[1:2])
print '\n view2'
mspl.mspl()
evaluate(mspl.Assment, realAssment)
print
mspl=MSPL(7,0.05,1.4,dataSet2,centroids2)
print '\n mspl'
mspl.mspl()
evaluate(mspl.Assment, realAssment)