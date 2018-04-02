#coding=utf-8
import numpy as ny
class MSPL:
    def __init__(self,centerNum,Lambda,mu,dataSet=[ny.mat([[]])]):
        self.dataSet=dataSet
        self.centerNum=centerNum
        self.Lambda=Lambda
        self.mu=mu
        self.viewNum=len(dataSet)
        self.dataNum=dataSet[0].shape[1]
        self.dims=[dataSet[i].shape[0] for i in range(self.viewNum)]
        self.weight=[[1]*self.dataNum for i in range(self.viewNum)]
        #聚类中心集初始化
        self.centroids=[ny.mat(ny.zeros((self.dims[i],self.centerNum))) for i in range(self.viewNum)]
        self.Assment=ny.mat(ny.zeros((centerNum,self.dataNum)))
    
    '''将第dataIndex个实例作为第cenIndex个聚类中心'''
    def setCentroid(self,dataIndex,cenIndex):
        for i in range(self.viewNum):
            self.centroids[i][:,cenIndex]=self.dataSet[i][:,dataIndex]
    
    '''欧式向量距离的平方函数'''
    def distEclud(self,vecA, vecB):
        return sum(ny.power(vecA - vecB, 2))
    
    '''第dataIndex个实例到第cenIndex个中心的加权距离的平方'''
    def distDataCen(self,dataIndex,cenIndex):
        Sum=0
        for i in range(self.viewNum):
            Sum+=self.weight[i][dataIndex]*self.distEclud(self.dataSet[i][:,dataIndex], self.centroids[i][:,cenIndex])[0,0]
        return Sum
    
    '''两个数据点的距离的平方'''
    def distDataDta(self,index1,index2):
        Sum=0
        for i in range(self.viewNum):
            Sum+=self.distEclud(self.dataSet[i][:,index1], self.dataSet[i][:,index2])[0,0]
        return Sum
    
    '''初始聚类中心生成函数'''        
    def Cent(self):
        index=int(ny.random.rand()*self.dataNum)
        self.setCentroid(index, 0)
        #num用于后续每次聚类中心选取的候选数据点数
        #num=min(max(self.centerNum*10,int(ny.sqrt(self.dataNum))),self.dataNum)
        for i in range(1,self.centerNum):
            '''在num个随机候选数据点中按离中心点集的最短距离作为多项分布参数，利用该分布生成随机数 b,将第b个实例作为新的聚类中心'''
            probList=[]
            for j in range(self.dataNum):
                #index=int(ny.random.rand()*self.dataNum)
                minDist=ny.inf
                for k in range(i):
                    dist=self.distDataCen(j, k)
                    if dist<minDist:
                        minDist=dist
                probList.append(minDist)
            #probList标准化
            probList=(ny.array(probList)/sum(probList)).tolist()
            index=ny.random.multinomial(1,probList).tolist().index(1)
            self.setCentroid(index, i)

    '''更新分配矩阵函数'''
    def updataAssment(self):
        changeFlag=False
        for i in range(self.dataNum):
            minDist=ny.inf
            minIndex=-1
            for j in range(self.centerNum):
                dist=self.distDataCen(i, j)
                if dist<minDist:
                    minDist=dist
                    minIndex=j
            if self.Assment[minIndex,i]!=1:
                changeFlag=True
                self.Assment[:,i]=0
                self.Assment[minIndex,i]=1
        return changeFlag
    
    '''计算第index个实例在第view视图下的损失'''
    def loss(self,view,index):
        #第index实例所属聚类中心号
        cenIndex=ny.array(self.Assment)[:,index].tolist().index(1)
        return self.distEclud(self.dataSet[view][:,index], self.centroids[view][:,cenIndex])[0,0]
        
    '''更新权重函数'''
    def updateWeight(self):
        e=ny.e
        allOneFlag=False
        for v in range(self.viewNum):
            for i in range(self.dataNum):
                l=self.loss(v,i)
                if l-1./self.Lambda>3:
                    self.weight[v][i]=0
                else:
                    self.weight[v][i]=(1+e**(-1./self.Lambda))/(1+e**(l-1./self.Lambda))
                if self.weight[v][i]!=1:
                    allOneFlag=True
        self.Lambda/=self.mu
        return allOneFlag
    
    '''更新中心矩阵函数'''
    def updateCentroids(self):
        for v in range(self.viewNum):
            W=ny.diag(ny.sqrt(self.weight[v]))
            self.centroids[v]=(self.dataSet[v]*W*W.T*self.Assment.T)*(self.Assment*W*W.T*self.Assment.T).I
 
    '''第二种中心矩阵更新函数'''
    def update2(self):
        for v in range(self.viewNum):
            self.centroids[v]*=0
            clusSize=[0]*self.centerNum
            for i in range(self.dataNum):
                for j in range(self.centerNum):
                    self.centroids[v][:,j]+=self.Assment[j,i]*self.weight[v][i]*self.dataSet[v][:,i]
                    clusSize[j]+=self.Assment[j,i]*self.weight[v][i]
        for v in range(self.viewNum):
            for l in range(self.centerNum):
                self.centroids[v][:,l]/=clusSize[l]

    def mspl(self):
        self.Cent()
        aFlag=self.updataAssment()
        wFlag=self.updateWeight()
        while aFlag or wFlag:
            self.updateCentroids()
            aFlag=self.updataAssment()
            wFlag=self.updateWeight()
        
'''            
view1=ny.mat([[0,0],[0,1],[1,0],[1,1],[4,4],[4,5],[5,4],[5,5]]).T
view2=ny.mat([[0,4,0],[1,4,0],[0,5,0],[1,5,0],[4,0,0],[4,1,0],[5,0,0],[5,1,0]]).T
mspl=MSPL(2,1,1.2,[view1,view2])
mspl.mspl()
print mspl.centroids[0]
print mspl.centroids[1],'\n'
print mspl.Assment,'\n'
print ny.mat(mspl.weight).T
mspl.setCentroid(0, 0)
mspl.Cent()
print mspl.centroids[0]
print mspl.centroids[1]
mspl.updataAssment()
print mspl.Assment
mspl.updateWeight()
print ny.mat(mspl.weight).T
mspl.updateCentroids()
print '\n'
print mspl.centroids[0]
print mspl.centroids[1],'\n'
mspl.update2()
print mspl.centroids[0]
print mspl.centroids[1]
'''