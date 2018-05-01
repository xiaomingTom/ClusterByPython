#coding:utf-8
import numpy as ny
#import scipy.io as sio
#import warnings
import time
#from Normalize import Normalize
from Evaluate import evaluate

class TW_kmeans:
    def __init__(self,centerNum,Lambda,Eta,lam,mu,dataSet=[ny.array([[]])]):
        self.setPara(centerNum,Lambda,Eta,lam,mu,dataSet)
        
    def setPara(self,centerNum,Lambda,Eta,lam,mu,dataSet=[ny.array([[]])]):
        #自步权重调整参数sλ
        self.lam=lam 
        #自步学习速率控制参数μ
        self.mu=mu 
        #聚类中心数k
        self.centerNum=centerNum
        #样本数据矩阵X
        self.dataSet=dataSet
        #视图个数
        self.viewNum=len(dataSet)
        #样本数量
        self.dataNum=dataSet[0].shape[1]
        #各视图维度
        self.dims=map(lambda x:x.shape[0],dataSet)
        #视图调整参数vλ
        self.Lambda=Lambda
        #特征权重调整参数η
        self.Eta=Eta
        #视图权重数组W
        self.W=ny.array([1./self.viewNum]*self.viewNum)
        #特征权重集合V,self.V[v]表示第v个视图下特征权重数组
        self.V=map(lambda x:ny.array([1./x]*x),self.dims)
        #分配矩阵B
        self.assment=ny.zeros((centerNum,self.dataNum))
        #自步权重数组矩阵
        self.weight=ny.ones((self.viewNum,self.dataNum))
    
    #获取初始中心矩阵函数(随机抽取k个点作为初始聚类中心)   
    def createCentroids(self):
        #聚类中心初始化
        self.centroids=map(lambda x:ny.zeros((x,self.centerNum)),self.dims)
        #随机获取k个数据点作为初始聚类中心
        for i in range(self.centerNum):
            #获取一个随机的样本点序号
            index=int(ny.random.rand()*self.dataNum)
            #将第index个样本点设置为聚类中心
            for v in range(self.viewNum):
                self.centroids[v][:,i]=self.dataSet[v][:,index]
    
    #===========================================
    #=设置聚类中心函数
    #=功能:将第dataIndex个样本点作为第cenIndex个聚类中心
    #===========================================
    def setCen(self,dataIndex,cenIndex):
        for i in range(self.viewNum):
            self.centroids[i][:,cenIndex]=self.dataSet[i][:,dataIndex]
    
    #================================================
    #获取概率离散的初始聚类中心函数
    #功能:选取k个数据点作为聚类中心，距离已有中心越远的点被选中概率越大
    #================================================
    def Cent(self):
        #聚类中心初始化
        self.centroids=map(lambda x:ny.zeros((x,self.centerNum)),self.dims)
        #随机获取一个样本点作为第一个聚类中心
        index=int(ny.random.rand()*self.dataNum)
        self.setCen(index,0)
        #各数据点到各聚类中心的最短距离数组
        minDist=[ny.inf]*self.dataNum
        dist=ny.array([0.]*self.dataNum)
        for i in range(1,self.centerNum):
            dist*=0
            #求各点到新聚类中心的距离
            for v in range(self.viewNum):
                dist+=((self.dataSet[v].T-self.centroids[v][:,i-1])**2).sum(1)
            #更新距聚类中心的最短距离数组
            minDist=map(lambda x,y:min(x,y),minDist,dist)
            #以最短距离数组为准生成多项分布参数
            probList=ny.array(minDist)/sum(minDist)
            #通过多项分布生成的结果获取新的聚类中心
            index=ny.random.multinomial(1,probList).tolist().index(1)
            self.setCen(index, i)
    
    #=======================================
    #求X矩阵与Y矩阵间的距离矩阵
    #=======================================
    def distXY(self,X,Y):
        x2=ny.tile((X.T**2).sum(1),(Y.shape[1],1)).T
        y2=ny.tile((Y**2).sum(0),(X.shape[1],1))
        return (x2+y2-2*X.T.dot(Y)).T
    
    #===================================================================
    #=获取第v个视图下各样本数据点到各聚类中心的距离矩阵函数
    #=原理参考:https://blog.csdn.net/Jiajing_Guo/article/details/62217564
    #====================================================================
    def matrDist(self,v):
        #特征加权数据源矩阵
        x=self.dataSet[v].T*self.V[v]**0.5
        #特征加权中心矩阵
        c=self.centroids[v]*ny.array([self.V[v]**0.5]).T
        #样本向量模平方和扩展矩阵
        x2=ny.tile((x**2).sum(1), (c.shape[1],1)).T
        #聚类中心向量模平方和扩展矩阵
        c2=ny.tile((c**2).sum(0),(x.shape[0],1))
        #返回加权距离矩阵
        return (x2+c2-2*x.dot(c))*ny.array([self.weight[v]]).T*self.W[v]
    
    #优化过的分配矩阵更新函数
    def updateA(self):
        #分配矩阵是否修改标注,False表示分配矩阵没有改动,目标函数局部收敛
        changeFlag=False;
        #获取各样本点到各样本中心的距离组合成的矩阵,即形成一个 n*k 的矩阵D,D(i,j)表示第i个样本点到第j个数据点的距离
        dist=ny.zeros((self.dataNum,self.centerNum))
        for v in range(self.viewNum):
            dist+=self.matrDist(v)
        for i in range(self.dataNum):
            #获取离第i个样本点最近的聚类中心的序号
            index=ny.argmin(dist[i])
            #判断第i个数据点的分配是否改变，更新分配矩阵
            if self.assment[index,i]!=1:
                changeFlag=True
                self.assment[:,i]=0
                self.assment[index,i]=1
        return changeFlag
    
    #分配矩阵更新函数
    def updateAssment(self):
        changeFlag=False;
        #对每个样本点进行重新分配
        for i in range(self.dataNum):
            #第i个样本点到k个聚类中心的距离数组
            dist=ny.array([0.]*self.centerNum)
            #计算dist数组
            for v in range(self.viewNum):
                dist+=((self.centroids[v].T-self.dataSet[v][:,i])**2*self.V[v]*self.W[v]*self.weight[v,i]).sum(1)
            #这里的index表示第i个样本点离第index个聚类中心最近
            index=ny.argmin(dist)
            #更新分配矩阵,并判断样本点的分配是否有改动
            if self.assment[index,i]!=1:
                changeFlag=True
                self.assment[:,i]=0
                self.assment[index,i]=1
        return changeFlag
    
    #===============================================
    #=聚类中心更新函数
    #=更新原理:
    #=1.符号表示:C 聚类中心矩阵, X 样本数据矩阵, B分配矩阵 , W 自步权重数组(向量)
    #=2.L=B.T · W.T (·乘这里表示,B.T矩阵的第i个行向量乘以W.T列向量的第i个元素得到新矩阵)
    #=3.L.sum(0) 表示 L矩阵纵向求和形成的数组(向量),即形成 [sum(L(i,1)),sum(L(i,2)),……]
    #=4.C=X*L / L.sum(0) 这里的除法表示X*L矩阵的每一列除以L.sum(0)向量对应的元素,
    #=        即X*L的第i个列向量除以L.sum(0)向量的第i个值
    #===============================================
    def updateCentroids(self):
        B=self.assment
        try:
            for v in range(self.viewNum):
                X=self.dataSet[v]
                W=ny.array([self.weight[v]])
                l=B.T*W.T
                self.centroids[v]=X.dot(l)/l.sum(0)
        except Exception,err:
            print l.sum(0)
            print B.sum(1)
            print v
            print err
            raise Exception
    
    def updateCen(self):
        t1,t2=0,0
        for v in range(self.viewNum):
            X=ny.mat(self.dataSet[v])
            #C=ny.mat(self.centroids[v])
            B=ny.mat(self.assment)
            tmp=time.time()
            D=((self.dataSet[v]-self.centroids[v].dot(self.assment))**2).sum(0)**0.5
            t1+=time.time()-tmp
            D=ny.mat(ny.diag(D))*0.5
            tmp=time.time()
            self.centroids[v]=ny.array(X*D*B.T*(B*D*B.T).I)
            t2+=time.time()-tmp
        print t1,t2
           
    def updateV(self):
        for v in range(self.viewNum):
            '''
            =原理:
            V[v]=E[v]/sum(E)
            E=e**( -sum( (X[v]-C[v]*B).F2·SW[v]*VW[v] , 行元素和  ) / η )
            '''
            E=ny.e**(-((self.dataSet[v]-self.centroids[v].dot(self.assment))**2*self.W[v]*self.weight[v]).sum(1)/self.Eta)
            self.V[v]=E/E.sum()
            
    
    def updateW(self):
        for v in range(self.viewNum):
            '''
            =原理:
            VW[v]=E[v]/sum(E)
            E[v]=e**(-sum( ((X[v]-C[v]*B).F2 · SW[v]).T · TW[v]) / Tλ)
            '''
            #print 'view',v,-(((self.dataSet[v]-self.centroids[v].dot(self.assment))**2*self.weight[v]).T*self.V[v]).sum()/self.Lambda
            self.W[v]=ny.e**(-( ((self.dataSet[v]-self.centroids[v].dot(self.assment))**2*self.weight[v]).T*self.V[v]).sum()/self.Lambda)
        self.W/=self.W.sum()
        
    
    '''更新样本权重函数'''
    def updateWeight(self):
        e=ny.e
        allOneFlag=False
        for v in range(self.viewNum):
            loss=((self.dataSet[v]-self.centroids[v].dot(self.assment)).T**2*self.V[v]*self.W[v]).sum(1)
            for i in range(self.dataNum):
                if loss[i]-1./self.lam>10:
                    self.weight[v][i]=1.0e-7
                else:
                    self.weight[v][i]=(1+e**(-1./self.lam))/(1+e**(loss[i]-1./self.lam))
                if self.weight[v][i]!=1.:
                    allOneFlag=True
        #self.lam/=self.mu
        return allOneFlag
    
    #kmeans聚类方法
    def kmeans(self):
        times=0
        self.W=ny.array([1.]*self.viewNum)
        self.V=map(lambda x:x/x[0],self.V)
        self.createCentroids()
        #self.Cent()
        #tmp=time.time()
        while self.updateA():
            #print time.time()-tmp
            self.updateCentroids()
            times+=1
        print times
        print self.means()
    
    def kcent(self):
        times=0
        self.W=ny.array([1.]*self.viewNum)
        self.V=map(lambda x:x/x[0],self.V)
        self.createCentroids()
        #self.Cent()
        while self.updateA():
            self.updateCen()
            times+=1
        print times
        print self.means()
    
    def tw_kmeans(self):
        times=0
        self.createCentroids()
        #self.Cent()
        while self.updateA():
            self.updateCentroids()
            self.updateV()
            self.updateW()
            times+=1
        print times
        
     
    def means(self):
        m=0
        for v in range(self.viewNum):
            m+=((self.dataSet[v]-self.centroids[v].dot(self.assment))**2).sum()*self.W[v]
        return m/self.viewNum/self.dataNum
    
    def mspl(self,lam):
        times=0
        self.W=ny.array([1.]*self.viewNum)
        self.V=map(lambda x:x/x[0],self.V)
        if not(lam is None):
            self.lam=lam
        self.createCentroids()
        #self.Cent()
        aFlag=self.updateA()
        print 'mean=',self.means()
        wFlag=self.updateWeight()
        while aFlag or wFlag:
            times+=1
            self.updateCentroids()
            aFlag=self.updateA()
            if not aFlag:
                self.lam/=self.mu
                wa=ny.array(self.weight)
                print times,'wa.mean=',wa.mean()
            if wFlag:
                wFlag=self.updateWeight()
        print 'means2=',self.means()
        print times
    
    def w_mspl(self,lam):
        times=0
        self.V=map(lambda x:x/x[0],self.V)
        if not(lam is None):
            self.lam=lam
        self.createCentroids()
        aFlag=self.updateA()
        self.updateW()
        print 'mean=',self.means()
        wFlag=self.updateWeight()
        while aFlag or wFlag:
            times+=1
            self.updateCentroids()
            aFlag=self.updateA()
            self.updateW()
            if not aFlag:
                self.lam/=self.mu
                wa=ny.array(self.weight)
                print times,'wa.mean=',wa.mean()
            if wFlag:
                wFlag=self.updateWeight()
        print 'means2=',self.means()
        print times
    
    def wkmeans(self):
        times=0
        self.V=map(lambda x:x/x[0],self.V)
        self.W=ny.array([1./self.viewNum]*self.viewNum)
        self.createCentroids()
        while self.updateA():
            self.updateW()
            self.updateCentroids()
            times+=1
        print times
        print self.means()
    
    def tw_sql(self,lam):
        times=0
        if not(lam is None):
            self.lam=lam
        self.createCentroids()
        #self.Cent()
        aFlag=self.updateA()
        print 'mean=',self.means()
        wFlag=self.updateWeight()
        while aFlag or wFlag:
            times+=1
            self.updateCentroids()
            self.updateV()
            self.updateW()
            aFlag=self.updateA()
            if wFlag:
                #wa=ny.array(self.weight)
                #print 'wa.mean=',wa.mean()
                wFlag=self.updateWeight()
        print 'means2=',self.means()
        print times


'''        
warnings.filterwarnings('error')  
matFile=sio.loadmat("D:\dataSet\handwritten.mat")
dataSet=[]
dataSet.append(matFile['mor'].T)
dataSet.append(matFile['fourier'].T)
dataSet.append(matFile['pixel'].T)
dataSet.append(matFile['kar'].T)
dataSet.append(matFile['profile'].T)
dataSet.append(matFile['zer'].T)
nor=Normalize()
dataSet=map(lambda x:ny.array(x*370),nor.normalize(dataSet))
gnd=matFile['gnd']
realAssment=[]
temp=ny.eye(10)
for i in range(gnd.shape[0]):
    realAssment.append(temp[gnd[i,0]].tolist())
tw=TW_kmeans(10,30*370**2,7*370**2,0.4,1.1,dataSet)
pur=[]
acc=[]
nmi=[]
for i in range(100):
    try:
        #tw.kmeans()
        tw.tw_kmeans()
        #tw.mspl(0.07)
        #tw.tw_sql(0.4)
        print tw.W
        p,a,n=evaluate(ny.mat(tw.assment), ny.mat(realAssment).T)
        pur.append(p)
        acc.append(a)
        nmi.append(n)
    except Exception,err:
        print err
        continue
print pur 
print acc
print nmi
print 'pur mean(std) max min std',('%.3f(%.3f)'%(ny.mean(pur),ny.std(pur))),max(pur),min(pur)
print 'acc mean(std) max min std',('%.3f(%.3f)'%(ny.mean(acc),ny.std(acc))),max(acc),min(acc)
print 'nmi mean(std) max min std',('%.3f(%.3f)'%(ny.mean(nmi),ny.std(nmi))),max(nmi),min(nmi)
'''