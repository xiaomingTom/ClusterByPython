#coding:utf-8
import numpy as ny
import scipy.io as sio
import warnings
from Evaluate import evaluate
from TW_SPL import TW_kmeans
from Normalize import Normalize

warnings.filterwarnings('error')
matFile=sio.loadmat("D:\dataSet\segment_uni.mat")
dataSet=ny.mat(matFile['X']).T
dataSet=[]
dataSet.append(matFile['X'][:,0:9].T)
dataSet.append(matFile['X'][:,9:19].T)
nor=Normalize()
dataSet=map(ny.mat,dataSet)
dataSet=map(lambda x:ny.array(x*6.8),nor.linerNormalize(dataSet))
#dataSet=map(lambda x:ny.array(x),nor.normalize(dataSet))
#dataSet=map(lambda x:x*3,dataSet)
#dataSet=map(lambda x:x*2,dataSet)
'''
x=0.2
dataSet[0]=dataSet[0]*x
dataSet[1]=dataSet[1]*(1-x)
'''
Y=matFile['Y']
realAssment=[]
etmp=ny.eye(7);
for i in range(Y.shape[0]):
    realAssment.append(etmp[Y[i,0]-1])
realAssment=ny.mat(realAssment).T
tw=TW_kmeans(7,380,2000,0.4,1.4,dataSet)
acc=[]
nmi=[]
pur=[]
for i in range(50):
    '''
    #tw.kmeans()
    tw.mspl(5)
    print tw.assment
    evaluate(ny.mat(tw.assment), realAssment)
    p,a,n=evaluate(ny.mat(tw.assment), realAssment)
    print p,a,n
    pur.append(p)
    acc.append(a)
    nmi.append(n)
    '''
    try:
        #tw.wkmeans()
        #tw.kcent()
        #tw.kmeans()
        #tw.tw_kmeans()
        tw.mspl(0.35)
        #pum.append(p)
        #acm.append(a)
        #nmm.append(n)
        #tw.tw_sql(0.7)
        print tw.W
        #print tw.V
        p,a,n=evaluate(ny.mat(tw.assment), realAssment)
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