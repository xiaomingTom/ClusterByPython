#coding=utf-8
#你好
import numpy
from Purity import Purity
from Hungary import Hungary
from NMI import NMI 
from DBI import DBI

def evaluate(clusterAssing,realAssment,dataMat,myCentroids):
    centerNum=numpy.shape(myCentroids)[1]
    purity=Purity()
    clusterVSet=purity.Divide(clusterAssing)
    realVSet=purity.Divide(realAssment)    
    purityTotal=0
    for i in range(centerNum):
        pur=purity.purity(realVSet, clusterVSet[i])
        print "pur",i,"=",pur
        purityTotal+=pur*len(clusterVSet[i])
    print 'the Purity=',purityTotal/(numpy.shape(dataMat)[1])
    probMatr=numpy.mat([[-purity.probIJ(clusterVSet[i], realVSet[j]) for j in range(centerNum)] 
                    for i in range(centerNum)])
    hungary=Hungary(probMatr)
    matchMatr=hungary.hungary()
    total=0
    for i in range(centerNum):
        for j in range(centerNum):
            total+=(-probMatr[i,j])*matchMatr[i][j]*len(clusterVSet[i])
    print 'the Accuracy=',total/(numpy.shape(dataMat)[1])
    nmi=NMI(clusterVSet,realVSet)
    print 'nmi=',nmi.nmi()
    #dbi=DBI(dataMat,myCentroids,clusterVSet)
    #print 'DBI=',dbi.dbi()