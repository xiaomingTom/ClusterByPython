#coding=utf-8
#原理参考https://blog.csdn.net/vernice/article/details/46467449
import numpy

#整理数据，封装成聚类集合集，即将同一类的元素的下标划分到一个set中，再将set组装成list返回
class Purity:
    def __init__(self):
        pass
    
    '''分配矩阵整理函数，将分配情况整理成一个划分list,这个划分list中的元素为一个个set集合,一个集合表示一个类'''
    def Divide(self,matr=numpy.mat([[]])):
        dim,num=numpy.shape(matr)
        clusterV=set()
        clusterVSet=[]
        for i in range(dim):
            vector=[0 for j in range(dim)]
            vector[i]=1
            clusterV=set()
            for k in range(num):
                if vector==numpy.array(matr)[:,k].tolist():
                    clusterV.add(k)
            clusterVSet.append(clusterV)
        return clusterVSet
    
    '''聚类i元素在类j中的概率'''
    def probIJ(self,clusterV=set(),realV=set()):
        return float(len(clusterV&realV))/len(clusterV)
    
    '''纯度函数，求出cluster聚类的纯度'''
    def purity(self,realVList=[set()],clusterV=set()):
        maxPro=0#最大概率
        for i in range(len(realVList)):
            Pro=float(len(clusterV&realVList[i]))/len(clusterV)
            if(maxPro< Pro):
                maxPro=Pro
        return maxPro
'''
a=numpy.mat([[1,0,0,0,0,1],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],])
print a
purity=Purity()
Set=purity.Divide(a)
b=set([0,3])
print Set[1]&b
print Set
print purity.purity(Set, b)
'''