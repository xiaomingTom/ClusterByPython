#coding=utf-8
import numpy

# 生成5*2的满足N(0,1)分布的样本矩阵
dataPart=2*numpy.random.randn(2,50)
dataSet=dataPart
#创建并配置分类矩阵
Assment=[]
for i in range(50):
    vector=[int(0) for j in range(4)]
    vector[0]=int(1)
    Assment.append(vector)
    
dataPart=3*numpy.random.randn(1,100)
datapart2=3*numpy.random.randn(1,100)+30
# hstack水平拼接矩阵,vstack垂直拼接矩阵
dataSet=numpy.hstack(( dataSet,numpy.vstack( (dataPart,datapart2) ) ))
for i in range(100):
    vector=[int(0) for j in range(4)]
    vector[1]=int(1)
    Assment.append(vector)

dataPart=4*numpy.random.randn(1,150)+30
datapart2=4*numpy.random.randn(1,150)
dataSet=numpy.hstack(( dataSet,numpy.vstack( (dataPart,datapart2) ) ))
for i in range(150):
    vector=[int(0) for j in range(4)]
    vector[2]=int(1)
    Assment.append(vector)

dataPart=5*numpy.random.randn(1,200)+30
datapart2=5*numpy.random.randn(1,200)+30
dataSet=numpy.hstack(( dataSet,numpy.vstack( (dataPart,datapart2) ) ))
for i in range(200):
    vector=[int(0) for j in range(4)]
    vector[3]=int(1)
    Assment.append(vector)

dataPart=12*numpy.random.randn(2,50)+15
dataSet=numpy.hstack((dataSet,dataPart))
dataSet=dataSet.T
Assment=numpy.mat(Assment)
'''
# shuffle乱序排列行
numpy.random.shuffle(dataSet)
'''
numpy.savetxt("d:/DataSet.txt",dataSet)
numpy.savetxt("d:/Assment.txt",Assment)
print dataSet
print numpy.shape(dataSet)
print '你好！'
