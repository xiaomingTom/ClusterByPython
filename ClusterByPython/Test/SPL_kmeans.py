#ecoding=utf-8 
import numpy
from matplotlib import pyplot as plt
from Test.Purity import Purity
from Test.Hungary import Hungary
 
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
 
'''获取初始聚类中心'''
def randCent(dataSet, k):
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
    '''       
    for j in range(dim):
        minJ = min(numpy.array(dataSet)[j])
        rangeJ = max(numpy.array(dataSet)[j])  - minJ 
        centroids[j,:] = minJ + rangeJ * numpy.random.rand(1,k)
    return centroids
    '''
    
'''自步学习kmeans聚类函数'''
def SPL_kMeans(dataSet, k, Lambda , mu , distMeas=distEclud, createCent=randCent):
    n = numpy.shape(dataSet)[1]
    #print 'n=',n
    #create mat to assign data points 
    clusterAssment = numpy.mat(numpy.zeros((k,n)))
    #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    weight=[]
    #print 'centroids.shape='+numpy.shape(centroids).__str__()
    #print 'clusterAssment.shape='+numpy.shape(clusterAssment).__str__()
    while clusterChanged:
        clusterChanged = False
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
        '''
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean 
        '''
        #update the weight of the samples
        weight=[]
        e=numpy.e
        for i in range(n):
            dist=distMeas(dataSet[:,i], centroids[:,numpy.array(clusterAssment)[:,i].tolist().index(1)]).tolist()[0][0]
            '''
            if 10*dist<1/Lambda:
                weight.append(1)
            else:
                weight.append( (1+e**(-1/Lambda)) / (1+e**(dist-1/Lambda)) )
            '''
            weight.append( (1+e**(-1/Lambda)) / (1+e**(dist-1/Lambda)) )
        #recalculate centroids
        W=numpy.diag(numpy.sqrt(weight))
        centroids = (dataSet*W*W.T*clusterAssment.T)*((clusterAssment*W*W.T*clusterAssment.T).I)
        #centroids = (dataSet*clusterAssment.T)*((clusterAssment*clusterAssment.T).I)
        Lambda/=mu
    return centroids, clusterAssment,weight
     
'''keans算法'''
def kMeans(dataSet, k,distMeas=distEclud, createCent=randCent):     
    n = numpy.shape(dataSet)[1]
    clusterAssment = numpy.mat(numpy.zeros((k,n)))
    #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
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
        '''
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean 
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

'''主函数'''      
def main():
    dataMat = loadDataSet("D:/DataSet.txt")
    #print 'dataMat.shape=' + numpy.shape(dataMat).__str__()
    centerNum = input('please input the number of the center:\n')
    Lambda = input('please input Lambda:\n')
    mu = input('please input mu(mu>1):\n')
    myCentroids,clustAssing=kMeans(dataMat, centerNum)
    show(dataMat, 4, myCentroids, clustAssing)
    myCentroids,clustAssing,weight= SPL_kMeans(dataMat,centerNum,Lambda,mu)
    print myCentroids
    print clustAssing
    print weight
    show(dataMat, 4, myCentroids, clustAssing)
    clusterAssing2=clustAssing[:,[i for i in range(0,numpy.shape(clustAssing)[1]-50)] ]
    realAssment = loadDataSet("d:/Assment.txt")
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
    
if __name__ == '__main__':
    main()