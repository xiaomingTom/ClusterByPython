#coding:utf-8
import numpy as ny
class Normalize:
    def __init__(self):
        pass
    
    def normalize(self,dataSet=[ny.mat([[]])]):
        norDataSet=[]
        for matr in dataSet:
            norMatr=ny.mat(ny.zeros(matr.shape))
            for i in range(matr.shape[0]):
                norMatr[i,:]=matr[i,:]/ny.linalg.norm(matr[i,:],2)
            norDataSet.append(norMatr)
        return norDataSet
        