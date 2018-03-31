#coding=utf-8
import numpy
from Test.SPL_kmeans import kMeans,SPL_kMeans
from Test.Purity import Purity
from Test.Hungary import Hungary
from Test.DBI import DBI
from Test.NMI import NMI
from Test.SPL_W_kmeans import SPL_W_kmeans

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
    dbi=DBI(dataMat,myCentroids,clusterVSet)
    print 'DBI=',dbi.dbi()

file=open('D:\dataSet\seeds_dataset.txt')
flag=True
dataMat=[]
Assment=[]
for line in file.readlines():
    if flag:
        flag=False
        continue
    curLine = line.strip().split()
    fltLine = map(float, curLine)
    if fltLine[-1]==1:
        Assment.append([1,0,0])
    elif fltLine[-1]==2:
        Assment.append([0,1,0])
    else:
        Assment.append([0,0,1])
    dataMat.append(fltLine[0:-1])
myCentroids,clusterAssing=kMeans(numpy.mat(dataMat).T, 3)
evaluate(clusterAssing, numpy.mat(Assment).T, numpy.mat(dataMat).T, myCentroids)
print '\n'
myCentroids2,clusterAssing2,samWeight=SPL_kMeans(numpy.mat(dataMat).T, 3, 0.25)
evaluate(clusterAssing2, numpy.mat(Assment).T, numpy.mat(dataMat).T, myCentroids2)
print samWeight
print '\n'
kmeans=SPL_W_kmeans(3,4,130,dataMat)
clusterAssing3,myCentroids3,attrWeight,samWeight=kmeans.kmeans()
#print numpy.mat(clusterAssing3)
evaluate(numpy.mat(clusterAssing3).T, numpy.mat(Assment).T, numpy.mat(dataMat).T, numpy.mat(myCentroids3).T)
print numpy.mat(attrWeight)
print samWeight
