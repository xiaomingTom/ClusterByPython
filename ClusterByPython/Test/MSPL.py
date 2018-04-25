#coding=utf-8
import numpy as ny
#import time

class MSPL:
    def __init__(self,centerNum,Lambda,mu,dataSet=[ny.mat([[]])],centroids=None):
        self.dataSet=dataSet
        self.centerNum=centerNum
        self.Lambda=Lambda
        self.mu=mu
        self.viewNum=len(dataSet)
        self.dataNum=dataSet[0].shape[1]
        self.dims=[dataSet[i].shape[0] for i in range(self.viewNum)]
        self.weight=[[1]*self.dataNum for i in range(self.viewNum)]
        #聚类中心集初始化
        if centroids!=None:
            '''centroids的深度拷贝'''
            self.centroids=[]
            for i in range(self.viewNum):
                self.centroids.append(ny.mat(centroids[i].tolist()))
        else:
            self.centroids=None
        #self.centroids=[ny.mat(ny.zeros((self.dims[i],self.centerNum))) for i in range(self.viewNum)]
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
        self.centroids=[ny.mat(ny.zeros((self.dims[i],self.centerNum))) for i in range(self.viewNum)]
        index=int(ny.random.rand()*self.dataNum)
        self.setCentroid(index, 0)
        probList=[ny.inf]*self.dataNum
        dist=ny.mat(ny.zeros((1,self.dataNum)))
        for i in range(1,self.centerNum):
            '''在num个随机候选数据点中按离中心点集的最短距离作为多项分布参数，利用该分布生成随机数 b,将第b个实例作为新的聚类中心'''
            dist*=0
            for v in range(self.viewNum):
                dist+=(ny.power(ny.tile(self.centroids[v][:,i-1], (1,self.dataNum))-self.dataSet[v],2)).sum(0)
            probList=map(lambda x,y:min(x,y),probList,dist.tolist()[0])
            '''
            for j in range(self.dataNum):
                probList[j]=min(probList[j],self.distDataCen(j, i-1))
            '''
            probList2=(ny.array(probList)/sum(probList)).tolist()
            index=ny.random.multinomial(1,probList2).tolist().index(1)
            self.setCentroid(index, i)
            
    def Cent2(self):
        index=int(ny.random.rand()*self.dataNum)
        self.setCentroid(index, 0)
        print 'c0=',index
        minDist=[ny.inf]*self.dataNum
        box=[0]*10
        box[index/200]=1
        for i in range(1,self.centerNum):
            '''在num个随机候选数据点中按离中心点集的最短距离作为多项分布参数，利用该分布生成随机数 b,将第b个实例作为新的聚类中心'''
            for j in range(self.dataNum):
                minDist[j]=min(minDist[j],self.distDataCen(j, i-1))
            index=minDist.index(max(minDist))
            self.setCentroid(index, i)
            print 'c',i,'=',index
            box[index/200]=box[index/200]+1
        print box
        
    def Cent3(self):
        for i in range(self.centerNum):
            index=int(ny.random.rand()*200)+i*200
            print index
            self.setCentroid(index, i)
        
    '''更新分配矩阵函数'''
    def updataAssment(self):
        changeFlag=False
        dist=ny.mat(ny.zeros((1,self.centerNum)))
        for i in range(self.dataNum):
            #minDist=ny.inf
            minIndex=-1
            dist*=0
            for v in range(self.viewNum):
                dist+=(ny.power(self.centroids[v]-self.dataSet[v][:,i],2)*self.weight[v][i]).sum(0)
                #dist+=(ny.power(ny.tile(self.dataSet[v][:,i],(1,self.centerNum))-self.centroids[v],2)*self.weight[v][i]).sum(0)
            minIndex=ny.argmin(dist)
            '''
            for j in range(self.centerNum):
                dist=self.distDataCen(i, j)
                if dist<minDist:
                    minDist=dist
                    minIndex=j
            '''
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
            loss=ny.power(self.dataSet[v]-self.centroids[v]*self.Assment,2).sum(0)
            #self.weight[v]=map(lambda x:0 if x-1./self.Lambda>4 else (1+e**(-1./self.Lambda))/(1+e**(x-1./self.Lambda)),ny.array(loss)[0])
            for i in range(self.dataNum):
                if loss[0,i]-1./self.Lambda>4:
                    self.weight[v][i]=0
                else:
                    self.weight[v][i]=(1+e**(-1./self.Lambda))/(1+e**(loss[0,i]-1./self.Lambda))
                if self.weight[v][i]!=1:
                    allOneFlag=True
        self.Lambda/=self.mu
        return allOneFlag
    
    '''更新中心矩阵函数'''
    def updateCentroids(self):
        for v in range(self.viewNum):
            W=ny.mat(ny.diag(ny.sqrt(self.weight[v])))
            self.centroids[v]=(self.dataSet[v]*W*W.T*self.Assment.T)*(self.Assment*W*W.T*self.Assment.T).I
 
    '''第二种中心矩阵更新函数'''
    def update2(self):
        B=ny.array(self.Assment)
        for v in range(self.viewNum):
            X=ny.array(self.dataSet[v])
            W=ny.array([self.weight[v]])
            l=B.T*W.T
            self.centroids[v]=ny.mat(X.dot(l)/l.sum(0))
                
    def means(self):
        m=0
        for v in range(self.viewNum):
            m+=ny.power(self.dataSet[v]-self.centroids[v]*self.Assment,2).sum()
        return m/self.dataNum/self.viewNum
     
    def createCentroids(self):
        self.centroids=[ny.mat(ny.zeros((self.dims[i],self.centerNum))) for i in range(self.viewNum)]
        for i in range(self.centerNum):
            index=int(ny.random.rand()*self.dataNum)
            for v in range(self.viewNum):
                self.centroids[v][:,i]=self.dataSet[v][:,index]
            
    def mspl(self):
        times=0
        if self.centroids==None:
            self.centroids=[ny.mat(ny.zeros((self.dims[i],self.centerNum))) for i in range(self.viewNum)]
            self.createCentroids()
        aFlag=self.updataAssment()
        self.Lambda=1./self.means()*3
        print 'mean=',self.means()
        wFlag=self.updateWeight()
        while aFlag or wFlag:
            times+=1
            self.update2()
            aFlag=self.updataAssment()
            if wFlag:
                wa=ny.array(self.weight)
                print 'wa.mean=',wa.mean()
                wFlag=self.updateWeight()
        print 'means2=',self.means()
        print times
    
    def Mspl(self,Lamdba=None):
        times=0
        if not(Lamdba is None):
            self.Lambda=Lamdba
        self.createCentroids()
        #self.Cent()
        aFlag=self.updataAssment()
        print 'mean=',self.means()
        wFlag=self.updateWeight()
        while aFlag or wFlag:
            times+=1
            self.update2()
            aFlag=self.updataAssment()
            if wFlag:
                wa=ny.array(self.weight)
                print 'wa.mean=',wa.mean()
                wFlag=self.updateWeight()
        print 'means2=',self.means()
        print times
       
    def mspl2(self):
        times=0
        if self.centroids==None:
            self.centroids=[ny.mat(ny.zeros((self.dims[i],self.centerNum))) for i in range(self.viewNum)]
            self.Cent()
        aFlag=self.updataAssment()
        self.Lambda=1./7
        for v in range(self.viewNum):
            loss=ny.power(self.dataSet[v]-self.centroids[v]*self.Assment,2).sum()/self.dataNum
            self.dataSet[v],self.centroids[v]= self.dataSet[v]*(8.0/loss)**0.5,self.centroids[v]*(8.0/loss)**0.5
        #print 'mean=',self.means()
        wFlag=self.updateWeight()
        while aFlag or wFlag:
            times+=1
            self.update2()
            aFlag=self.updataAssment()
            if wFlag:
                wa=ny.array(self.weight)
                print 'wa.mean=',wa.mean()
                wFlag=self.updateWeight()
        print 'means2=',self.means()
        print times