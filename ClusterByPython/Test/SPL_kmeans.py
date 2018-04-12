#coding=utf-8 
import numpy
import time
from matplotlib import pyplot as plt

'''数据加载函数''' 
def loadDataSet(fileName):
    dataMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split()
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    dataMat=numpy.mat(dataMat)
    return dataMat.T
     
'''欧式距离计算函数'''
def distEclud(vecA, vecB):
    return numpy.sqrt(sum(numpy.power(vecA - vecB, 2)))

'''获取随机初始聚类中心'''
def randCent(dataSet,k):
    dim , dataNum= numpy.shape(dataSet)
    centroids = numpy.mat(numpy.zeros((dim,k)))
    centroids[:,0]=dataSet[:,int(numpy.random.rand() * dataNum)]       
    for j in range(dim):
        minJ = min(numpy.array(dataSet)[j])
        rangeJ = max(numpy.array(dataSet)[j])  - minJ 
        centroids[j,:] = minJ + rangeJ * numpy.random.rand(1,k)
    return centroids

'''获取离散程度尽量大的初始聚类中心'''    
def disperseCent(dataSet,k):
    dim , dataNum= numpy.shape(dataSet)
    centroids = numpy.mat(numpy.zeros((dim,k)))
    centroids[:,0]=dataSet[:,int(numpy.random.rand() * dataNum)]
    for i in range(1,k):
        maxDist=0
        maxIndex=-1
        for j in range(dataNum):
            dist=0
            for g in range(i):
                dist+=distEclud(dataSet[:,j], centroids[:,g])
            if maxDist<dist:
                maxDist=dist
                maxIndex=j
        centroids[:,i]=dataSet[:,maxIndex]
    return centroids

def cent(dataSet,k):
    dim = numpy.shape(dataSet)[0]
    centroids = numpy.mat(numpy.zeros((dim,k)))
    for i in range(k):
        index=int(numpy.random.rand()*200)+i*200
        centroids[:,i]=dataSet[:,index]
    return centroids

def Cent2(dataSet,k):
    dim , dataNum= numpy.shape(dataSet)
    centroids = numpy.mat(numpy.zeros((dim,k)))
    centroids[:,0]=dataSet[:,int(numpy.random.rand() * dataNum)]
    for i in range(1,k):
        distSum=0
        maxDist=0
        maxIndex=-1
        for j in range(dataNum):
            dist=0
            minDist=numpy.inf
            for g in range(i):
                dist=distEclud(dataSet[:,j], centroids[:,g])[0,0]
                if dist<minDist:
                    minDist=dist
            if minDist>maxDist:
                maxDist,maxIndex=minDist,j
        centroids[:,i]=dataSet[:,maxIndex]
    return centroids       

'''获取初始聚类中心(最优方法)'''
def Cent(dataSet, k):
    dim , dataNum= numpy.shape(dataSet)
    centroids = numpy.mat(numpy.zeros((dim,k)))
    centroids[:,0]=dataSet[:,int(numpy.random.rand() * dataNum)]
    for i in range(1,k):
        distSum=0
        probList=[]
        for j in range(dataNum):
            dist=0
            minDist=numpy.inf
            for g in range(i):
                dist=distEclud(dataSet[:,j], centroids[:,g])[0,0]
                if dist<minDist:
                    minDist=dist
            distSum+=minDist
            probList.append(minDist)
        probList=[probList[s]/distSum for s in range(len(probList))]
        Index=numpy.random.multinomial(1,probList) 
        centroids[:,i]=dataSet[:,Index.tolist().index(1)]
    return centroids       
    
'''自步学习kmeans聚类函数'''
def SPL_kMeans(dataSet, k, Lambda  , distMeas=distEclud, createCent=disperseCent):
    n = numpy.shape(dataSet)[1]
    #create mat to assign data points 
    clusterAssment = numpy.mat(numpy.zeros((k,n)))
    #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    weight=[]
    flag=True
    while clusterChanged:
        clusterChanged = False
        for i in range(n):#for each data point assign it to the closest centroid
            minDist = numpy.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[:,j],dataSet[:,i])
                if distJI < minDist:
                    minDist = distJI; minIndex = j 
            if clusterAssment[minIndex,i] != 1:
                clusterChanged = True
                clusterAssment[:,i] = 0
                clusterAssment[minIndex,i]=1
        '''
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean 
        '''
        dist=[]
        if flag:
            flag=False
            for g in range(n):
                dist.append(distMeas(dataSet[:,g],centroids*clusterAssment[:,g])[0,0]**2)
            #print dist
            print 'mean=',numpy.mean(dist)
            print 'variance=',numpy.var(dist)
        #update the weight of the samples
        weight=[]
        e=numpy.e
        for i in range(n):
            dist=distMeas(dataSet[:,i], centroids[:,numpy.array(clusterAssment)[:,i].tolist().index(1)]).tolist()[0][0]
            if dist**2-1/Lambda>3:
                weight.append(0)
            else:
                weight.append( (1+e**(-1/Lambda)) / (1+e**(dist**2-1/Lambda)) )
        #recalculate centroids
        W=numpy.diag(numpy.sqrt(weight))
        centroids = (dataSet*W*W.T*clusterAssment.T)*((clusterAssment*W*W.T*clusterAssment.T).I)
    return centroids, clusterAssment,weight
     
'''keans算法'''
def kMeans(dataSet, k,centroids=None,distMeas=distEclud, createCent=cent):     
    n = numpy.shape(dataSet)[1]
    clusterAssment = numpy.mat(numpy.zeros((k,n)))
    #to a centroid, also holds SE of each point
    if centroids==None:
        centroids = createCent(dataSet, k)
    clusterChanged = True
    times=0
    while clusterChanged:
        clusterChanged = False
        print times
        times+=1
        for i in range(n):#for each data point assign it to the closest centroid
            minDist = numpy.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[:,j],dataSet[:,i])
                #print distJI.tolist()
                if distJI < minDist:
                    minDist = distJI; minIndex = j 
            if clusterAssment[minIndex,i] != 1:
                clusterChanged = True
                clusterAssment[:,i] = 0
                clusterAssment[minIndex,i]=1
        #更新中心矩阵
        '''
        clusSize=[0]*k
        centroids*=0
        for i in range(n):
            for j in range(k):
                centroids[:,j]+=dataSet[:,i]*clusterAssment[j,i]
                clusSize[j]+=clusterAssment[j,i]
        for i in range(k):
            centroids[:,i]/=clusSize[i]
        '''
        centroids = (dataSet*clusterAssment.T)*((clusterAssment*clusterAssment.T).I)
    return centroids, clusterAssment

'''画图函数'''    
def show(dataSet, k, centroids, clusterAssment):
    numSamples = dataSet.shape[1]  
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr'] 
    for i in xrange(numSamples):  
        markIndex = int( numpy.array(clusterAssment)[:,i].tolist().index(1) )  
        plt.plot(dataSet[0, i], dataSet[1, i], mark[markIndex])  
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']  
    for i in range(k):  
        plt.plot(centroids[0, i], centroids[1, i], mark[i], markersize = 12)  
    plt.show()
