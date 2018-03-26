#ecoding=utf-8
from Test.SPL_kmeans import *
from Test.Purity import Purity
from Test.Hungary import Hungary
from Test.DBI import DBI
from Test.NMI import NMI

'''主函数'''      
def main():
    dataMat = loadDataSet("D:/DataSet.txt")
    centerNum = input('please input the number of the center:\n')
    Lambda = input('please input Lambda:\n')
    mu = input('please input mu(mu>1):\n')
    '''-
    myCentroids,clustAssing=kMeans(dataMat, centerNum)
    show(dataMat, 4, myCentroids, clustAssing)
    '''
    myCentroids,clustAssing,weight= SPL_kMeans(dataMat,centerNum,Lambda,mu)
    print myCentroids
    print clustAssing
    print weight
    show(dataMat, 4, myCentroids, clustAssing)
    
    clusterAssing2=clustAssing[:,[i for i in range(0,numpy.shape(clustAssing)[1]-50)] ]
    realAssment = loadDataSet("c:/Assment.txt")
    purity=Purity()
    clusterVSet=purity.Divide(clusterAssing2)
    realVSet=purity.Divide(realAssment)    
    purityTotal=0
    for i in range(centerNum):
        pur=purity.purity(realVSet, clusterVSet[i])
        print "pur",i,"=",pur
        purityTotal+=pur*len(clusterVSet[i])
    print 'the Purity=',purityTotal/(numpy.shape(dataMat)[1]-50)
    probMatr=numpy.mat([[-purity.probIJ(clusterVSet[i], realVSet[j]) for j in range(centerNum)] 
                        for i in range(centerNum)])
    print probMatr
    hungary=Hungary(probMatr)
    matchMatr=hungary.hungary()
    print matchMatr
    total=0
    for i in range(centerNum):
        for j in range(centerNum):
            total+=(-probMatr[i,j])*matchMatr[i][j]*len(clusterVSet[i])
    print 'the Accuracy=',total/(numpy.shape(dataMat)[1]-50)
    nmi=NMI(clusterVSet,realVSet)
    print 'nmi=',nmi.nmi()
    clusterVSet=purity.Divide(clustAssing)
    dbi=DBI(dataMat,myCentroids,clusterVSet)
    print 'DBI=',dbi.dbi()
    
    
    
if __name__ == '__main__':
    main()