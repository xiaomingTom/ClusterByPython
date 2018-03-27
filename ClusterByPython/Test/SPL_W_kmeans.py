#coding=utf-8
import numpy
'''自步学习特征子空间kmeans算法'''
class SPL_W_kmeans:
    def __init__(self,DataMat=[[]],centerNum,Lambda,mu,eta):
        self.DataMat=numpy.mat(DataMat)
        self.centerNum=centerNum
        self.Lambda=Lambda
        self.mu=mu
        self.eta=eta
    
    '''欧式距离函数'''
    def dist(self,x,y):
        return numpy.sqrt(x**2-y**2)
    
    '''更新分配矩阵assment'''