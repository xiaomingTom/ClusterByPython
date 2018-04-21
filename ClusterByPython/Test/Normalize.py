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
                if matr[i,:].sum()!=0:
                    norMatr[i,:]=matr[i,:]/ny.linalg.norm(matr[i,:],2)
            norDataSet.append(norMatr)
        return norDataSet
    
    def normalize2(self,dataSet=[ny.mat([[]])]):
        norDataSet=[]
        s=0
        for matr in dataSet:
            norMatr=ny.mat(ny.zeros(matr.shape))
            for i in range(matr.shape[0]):
                if matr[i,:].sum()!=0:
                    norMatr[i,:]=matr[i,:]/ny.linalg.norm(matr[i,:],2)
                else:
                    norMatr[i,:]=1./matr.shape[1]
                    print s,'视图',i,'特征全部为0，无法对该特征归一'
            dim,num=norMatr.shape
            norDataSet.append(norMatr*ny.power(float(num)/dim,0.5))
            s+=1
        return norDataSet
    
    def linerNormalize(self,dataSet=[ny.mat([[]])]):
        norDataSet=[]
        for matr in dataSet:
            norMatr=ny.mat(ny.zeros(matr.shape))
            for i in range(matr.shape[0]):
                norMatr[i,:]=(matr[i,:]-matr[i,:].min())*1.0/(matr[i,:].max()-matr[i,:].min())
            norDataSet.append(norMatr)
        return norDataSet
    
    def zeroNormalize(self,dataSet=[ny.mat([[]])]):
        norDataSet=[]
        for matr in dataSet:
            norMatr=ny.mat(ny.zeros(matr.shape))
            for i in range(matr.shape[0]):
                norMatr[i,:]=(matr[i,:]-matr[i,:].mean())/matr[i,:].var()
            norDataSet.append(norMatr)
        return norDataSet
    
        