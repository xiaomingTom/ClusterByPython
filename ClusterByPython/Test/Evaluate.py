#coding=utf-8
#你好
import numpy
from Purity import Purity
from Hungary import Hungary
from NMI import NMI 
#from DBI import DBI

def evaluate(clusterAssing,realAssment):
    centerNum,dataNum=clusterAssing.shape
    purity=Purity()
    clusterVSet=purity.Divide(clusterAssing)
    realVSet=purity.Divide(realAssment)    
    purityTotal=0
    for i in range(centerNum):
        pur=purity.purity(realVSet, clusterVSet[i])
        #print "pur",i,"=",pur
        purityTotal+=pur*len(clusterVSet[i])
    #print 'the Purity=',purityTotal/dataNum
    probMatr=numpy.mat([[-purity.probIJ(clusterVSet[i], realVSet[j]) for j in range(centerNum)] 
                    for i in range(centerNum)])
    hungary=Hungary(probMatr)
    matchMatr=hungary.hungary()
    total=0
    for i in range(centerNum):
        for j in range(centerNum):
            total+=(-probMatr[i,j])*matchMatr[i][j]*len(clusterVSet[i])
    #print 'the Accuracy=',total/dataNum
    nmi=NMI(clusterVSet,realVSet)
    n=nmi.nmi()
    #print 'nmi=',n
    return purityTotal/dataNum,total/dataNum,n
    #dbi=DBI(dataMat,myCentroids,clusterVSet)
    #print 'DBI=',dbi.dbi()