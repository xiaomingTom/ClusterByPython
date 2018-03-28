#coding=utf-8
import xlrd
import numpy
from SPL_W_kmeans import SPL_W_kmeans
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
dataMat=rowLines

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
Lambda = input('please input Lambda(Lambda>0):\n')
eta=input('please input eta(eta>0):\n')
kmeans=SPL_W_kmeans(centerNum,Lambda,eta,dataMat)
clusterAssing,myCentroids,attrWeight,samWeight=kmeans.kmeans()
print 'centoids=\n',myCentroids
print 'attrWeight=\n',attrWeight
print 'samWeight=\n',samWeight
print 'clusetAssing=\n',clusterAssing
print 'samWeight=\n',samWeight
purity=Purity()
clusterVSet=purity.Divide(numpy.mat(clusterAssing).T)
realVSet=purity.Divide(realAssment)    
purityTotal=0
for i in range(centerNum):
    pur=purity.purity(realVSet, clusterVSet[i])
    print "pur",i,"=",pur
    purityTotal+=pur*len(clusterVSet[i])
print 'the Purity=',purityTotal/(len(dataMat))
probMatr=numpy.mat([[-purity.probIJ(clusterVSet[i], realVSet[j]) for j in range(centerNum)] 
                    for i in range(centerNum)])
hungary=Hungary(probMatr)
matchMatr=hungary.hungary()
total=0
for i in range(centerNum):
    for j in range(centerNum):
        total+=(-probMatr[i,j])*matchMatr[i][j]*len(clusterVSet[i])
print 'the Accuracy=',total/(len(dataMat))
nmi=NMI(clusterVSet,realVSet)
print 'nmi=',nmi.nmi()
dbi=DBI(numpy.mat(dataMat).T,numpy.mat(myCentroids).T,clusterVSet)
print 'DBI=',dbi.dbi()
