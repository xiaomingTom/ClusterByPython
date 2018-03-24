#coding=utf-8
#原理参考:https://blog.csdn.net/sinat_33363493/article/details/52496011
import numpy

class DBI:
    def __init__(self,dataSet=numpy.mat([[]]), centroid=numpy.mat([[]]),clusterAssmemt=[set()]):
        self.dataSet=dataSet
        self.centroid=centroid
        self.clusterAssment=clusterAssmemt
    
    #F范数计算函数
    def Frobenius(self,matr=numpy.mat([[]])):
        return numpy.trace(matr.T*matr)
    
    '''欧式距离计算函数'''
    def distEclud(self,vecA, vecB):
        return numpy.sqrt(sum(numpy.power(vecA - vecB, 2)))
    
    '''第i个类的紧密性计算函数(CP)'''
    def CP(self,i):
        distance=0#总距离
        for j in self.clusterAssment[i]:
            distance+=self.distEclud(self.dataSet[:,j], self.centroid[:,i])
        return distance[0,0]/len(self.clusterAssment)

    def dbi(self):
        SumDist=0
        for i in range(numpy.shape(self.centroid)[1]):
            MaxDist=0
            for j in range(numpy.shape(self.centroid)[1]):
                if i==j:
                    continue
                dist=(self.CP(i)+self.CP(j))/self.distEclud(self.centroid[:,i], self.centroid[:,j])
                if MaxDist<dist:
                    MaxDist=dist
            SumDist+=MaxDist
        return (SumDist/numpy.shape(self.centroid)[1])[0,0]

'''
dataSet=numpy.mat([[1,2,3,4],[5,6,7,8],[1,5,4,9],[7,5,2,1]])
centroid=(numpy.mat([[0,0,0,0],[1,1,1,1]])).T
clusterAssment=[{0,2},{1,3}]
dbi=DBI(dataSet,centroid,clusterAssment)


print dbi.CP(0)
print dbi.CP(1)
print dbi.dbi()
'''
        