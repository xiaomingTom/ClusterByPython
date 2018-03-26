#ecoding=utf-8
import xlrd
from SPL_kmeans import *
from Test.Purity import Purity
from Test.Hungary import Hungary
from Test.DBI import DBI
from Test.NMI import NMI

xlsFile=xlrd.open_workbook('d:/Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.xls')
dataTable=xlsFile.sheets()[1]
rowLines=[]
#获取标签信息
lable=dataTable.col_values(5)
#获取数据信息
for i in range(1,dataTable.nrows):
    rowLines.append(dataTable.row_values(i)[0:5])
dataMat=numpy.mat(rowLines).T

#将标签信息整理成分类矩阵
realAssment=[]
for i in range(len(lable)):
    if lable[i]=='very_low':
        realAssment.append([1,0,0,0])
    elif lable[i]=='Low':
        realAssment.append([0,1,0,0])
    elif lable[i]=='Middle':
        realAssment.append([0,0,1,0])
    else:
        realAssment.append([0,0,0,1])
realAssment=numpy.mat(realAssment).T


centerNum = input('please input the number of the center:\n')
Lambda = input('please input Lambda:\n')
mu = input('please input mu(mu>1):\n')
#myCentroids,clusterAssing,weight= SPL_kMeans(dataMat,centerNum,Lambda,mu)
myCentroids,clusterAssing=kMeans(dataMat, centerNum)
show(dataMat, 4, myCentroids, clusterAssing)
#print weight
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
print probMatr
hungary=Hungary(probMatr)
matchMatr=hungary.hungary()
print matchMatr
total=0
for i in range(centerNum):
    for j in range(centerNum):
        total+=(-probMatr[i,j])*matchMatr[i][j]*len(clusterVSet[i])
print 'the Accuracy=',total/(numpy.shape(dataMat)[1])
nmi=NMI(clusterVSet,realVSet)
print 'nmi=',nmi.nmi()
dbi=DBI(dataMat,myCentroids,clusterVSet)
print 'DBI=',dbi.dbi()
