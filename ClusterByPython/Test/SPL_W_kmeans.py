#coding=utf-8
import numpy

'''自步学习特征子空间kmeans算法'''
class SPL_W_kmeans:
    def __init__(self,centerNum,Lambda,eta,dataMat=[[]]):
        self.dataMat=dataMat#数据矩阵
        self.centerNum=centerNum#聚类中心数
        self.Lambda=Lambda
        self.eta=eta
        self.dataNum=len(dataMat)
        self.dim=len(dataMat[0])#数据维度和数量
        self.attrWeight=[[1.0/self.dim for j in range(self.dim)]for i in range(centerNum)]#特征子空间权重
        self.Assment=numpy.zeros((self.dataNum,centerNum)).tolist()
    
    '''欧式距离函数'''
    def dist(self,x,y):
        return numpy.sqrt((x-y)**2)
    
    '''向量欧式距离算法函数'''
    def distVector(self,vecA,vecB):
        return numpy.sqrt(sum(numpy.power(vecA - vecB, 2)))
    
    '''获取初始聚类中心'''
    def Cent(self):
        self.centroids = numpy.zeros((self.centerNum,self.dim)).tolist()
        self.centroids[0]=self.dataMat[int(numpy.random.rand() * self.dataNum)]
        for i in range(1,self.centerNum):
            distSum=0
            probList=[]
            for j in range(self.dataNum):
                dist=0
                minDist=numpy.inf
                for g in range(i):
                    dist=self.distVector(numpy.array(self.dataMat[j]), numpy.array(self.centroids[g]))
                    if dist<minDist:
                        minDist=dist
                distSum+=minDist
                probList.append(minDist)
            probList=[probList[s]/distSum for s in range(len(probList))]
            probList=(numpy.array(probList)).tolist()
            #用距离比作参数产生一个满足多项分布的随机数，将数据集中下标为这个随机数的点作为新的聚类中心
            Index=numpy.random.multinomial(1,probList) 
            self.centroids[i]=list(self.dataMat[Index.tolist().index(1)])
        
        
    '''更新分配矩阵assment函数'''
    def updateAssment(self):
        flag=False
        for i in range(self.dataNum):
            minDist=numpy.inf
            minIndex=-1
            for j in range(self.centerNum):
                dist=0
                for k in range(self.dim):
                    dist+=self.attrWeight[j][k]*(self.dist(self.centroids[j][k],self.dataMat[i][k])**2)
                if dist<minDist:
                    minDist=dist
                    minIndex=j
            if self.Assment[i][minIndex]!=1:
                self.Assment[i]=[0]*self.centerNum#len(self.Assment[i])
                self.Assment[i][minIndex]=1    
                flag=True
        return flag
  
    '''更新自步权重向量samWeight函数'''
    def updataSamWeight(self):
        self.samWeight=[]
        e=numpy.e
        for i in range(self.dataNum):
            dist=0
            cenIndex=self.Assment[i].index(1)#第i个实例所属聚类的类号
            for k in range(self.dim):
                dist+=self.attrWeight[cenIndex][k]*self.dist(self.centroids[cenIndex][k],self.dataMat[i][k])
            if dist**2-1/self.Lambda>3:
                self.samWeight.append(0)
            else:
                self.samWeight.append( (1+e**(-1/self.Lambda)) / (1+e**(dist**2-1/self.Lambda)) )
        
    '''更新聚类中心矩阵函数'''
    def updataCentroids(self):
        '''
        print '\n\n'
        print 'Assment[-1]',numpy.array(self.Assment)[:,2]
        print 'sampleWeight[-1]',numpy.array(self.samWeight)[self.dim-1:-1]
        '''
        for l in range(self.centerNum):
            for j in range(self.dim):
                self.centroids[l][j]=(sum([self.Assment[s][l]*self.samWeight[s]*
                    self.dataMat[s][j] for s in range(self.dataNum)])/
                    sum([self.Assment[q][l]*self.samWeight[q] for q in range(self.dataNum)]))

    '''更新特征子空间权重'''
    def updateAttrWeight(self):
        Elj=numpy.zeros((self.centerNum,self.dim)).tolist()
        for l in range(self.centerNum):
            for j in range(self.dim):
                Elj[l][j]=(sum([self.Assment[i][l]*self.samWeight[i]*
                    self.dist(self.dataMat[i][j],self.centroids[l][j])**2 for i in range(self.dataNum)]))
        e=numpy.e
        for l in range(self.centerNum):
            elist=[]
            for j in range(self.dim):
                elist.append(e**(-Elj[l][j]/self.eta))
            self.attrWeight[l]=(numpy.array(elist)/sum(elist)).tolist()


    def kmeans(self):
        self.Cent()
        while self.updateAssment(): 
            self.updataSamWeight()
            self.updataCentroids()
            self.updateAttrWeight()
        return self.Assment,self.centroids,self.attrWeight,self.samWeight
    
'''         
dataMat=[[0,0],[0,1],[1,0],[1,1],[4,4],[4,5],[5,4],[5,5]]
kmeans=SPL_W_kmeans(2,1,1,dataMat)
kmeans.Cent()
kmeans.updateAssment()
kmeans.updataSamWeight()
print numpy.mat(kmeans.Assment)
print numpy.mat(kmeans.centroids)
print numpy.mat(kmeans.attrWeight)
print kmeans.samWeight
kmeans.updataCentroids()
print numpy.mat(kmeans.centroids)
kmeans.updateAttrWeight()
kmeans.kmeans()
print numpy.mat(kmeans.Assment)
print numpy.mat(kmeans.centroids)
print numpy.mat(kmeans.attrWeight)
print kmeans.samWeight
'''