#coding:utf-8
import numpy as ny
import scipy.io as sio
import warnings
#import time
from Normalize import Normalize
from Evaluate import evaluate
from TW_SPL import TW_kmeans

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
dataSet=map(ny.mat,dataSet)
#dataSet=map(lambda x:ny.array(x*6),nor.rowNormalize(dataSet))
dataSet=map(lambda x:ny.array(x*5),nor.linerNormalize(dataSet))
'''
w=ny.array([1./6**0.5,1./76**0.5,1./240**0.5,1./64**0.5,1./216**0.5,1./47**0.5])
w/=w.sum()
print w
dataSet=map(lambda x,y:x*y,dataSet,w)
'''
gnd=matFile['gnd']
realAssment=[]
temp=ny.eye(10)
for i in range(gnd.shape[0]):
    realAssment.append(temp[gnd[i,0]].tolist())
tw=TW_kmeans(10,12000*25,7,0.15,1.6,dataSet)
'''
x=ny.array([[1,2],[3,4],[5,6],[7,8]])
c=ny.array([[2,4,7],[1,5,6]])
print tw.matrDist(x, c)
'''
pur=[]
acc=[]
nmi=[]
for i in range(10):
    
    try:
        #tw.kcent()
        tw.kmeans()
        #tw.tw_kmeans()
        #tw.mspl(0.5)
        #tw.w_mspl(0.5)
        #tw.tw_sql(0.45)
        #tw.wkmeans()
        #print tw.W
        #print tw.V
        p,a,n=evaluate(ny.mat(tw.assment), ny.mat(realAssment).T)
        print 'purity=',p,'acc=',a,'nmi=',n
        print 
        pur.append(p)
        acc.append(a)
        nmi.append(n)
    except Exception,err:
        print err
        continue
print pur 
print acc
print nmi
print 'pur mean(std) max min std',('%.3f(%.3f)'%(ny.mean(pur),ny.std(pur))),max(pur),min(pur)
print 'acc mean(std) max min std',('%.3f(%.3f)'%(ny.mean(acc),ny.std(acc))),max(acc),min(acc)
print 'nmi mean(std) max min std',('%.3f(%.3f)'%(ny.mean(nmi),ny.std(nmi))),max(nmi),min(nmi)