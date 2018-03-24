#coding=utf-8
#原理参考http://www.cnblogs.com/chenyg32/p/3293247.html
import numpy

class Hungary:
    def __init__(self,matr=numpy.mat([])):
        self.matr=matr
        #m行,n列,要求m<=n
        self.m,self.n=numpy.shape(matr)
        
    def setMatr(self,matr):
        self.matr=matr
        self.m,self.n=numpy.shape(matr)
        
    def getMatr(self):
        return self.matr
        
    #归约函数
    
    '''归约函数'''
    def reduce(self):
        #行归约
        for i in range(self.m):
            self.matr[i]-=min(self.matr.tolist()[i])
        #列归约
        for i in range(self.n):
            self.matr[:,i]-=min(self.matr[:,i].tolist())
    
    '''试指派函数'''
    def tryMatch(self):
        rowFlag=[False for i in range(self.m)]#行划线flag
        colFlag=[False for i in range(self.n)]#列划线flag
        i=0#现已知独立0元素个数
        flag=True
        zeroSet=[]#独立零元素list,集合元素以(行号,列号)形式插入
        while i<self.m and flag:
            r=-1
            c=-1
            minZeros=self.n+1#行(列)最少含未划线0数
            minZeroLoc=(-1,-1)#含有最少未划线0元素的行()的列最后一个零元素所在位置
            flag=False#该次循环是否有行或列被划线标志
            #查找所有未划线行中含有最少未划线0元素的行，并对该0元素所在行和列划线等操作
            for j in range(self.m):
                #j行已被划线不做考虑，跳过
                if rowFlag[j]:
                    continue
                count=0#j行含0的元素个数
                
                #计算j行未被划线的0的个数
                for k in range(self.n):
                    if self.matr[j,k]==0 and not colFlag[k]:
                        count+=1
                        r,c=j,k
                #若j行含有未被划线的零元素数小于minZeros，更新minZeros和minZeroLoc
                if count!=0 and count<minZeros:
                    minZeros=count
                    minZeroLoc=(r,c)
            #查找所有未划线列中只含一个0元素的列，并对该0元素所在行和列划线等操作
            for j in range(self.n):
                #j列已被划线不做考虑，跳过
                if colFlag[j]:
                    continue
                count=0#j列含0的元素个数
                #计算j列未被划线的0的个数
                for k in range(self.m):
                    if self.matr[k,j]==0 and not rowFlag[k]:
                        count+=1
                        r,c=k,j
                #若j列含有未被划线的零元素数小于minZeros，更新minZeros和minZeroLoc
                if count!=0 and count<minZeros:
                    minZeros=count
                    minZeroLoc=(r,c)
            #若有最少未划线的零元素的行(列)，则对该零元素所在行和列划线，flag置True,minZeroLoc插入到zeroSet
            if minZeros<self.n+1:
                rowFlag[minZeroLoc[0]]=True
                colFlag[minZeroLoc[1]]=True
                i+=1
                flag=True
                zeroSet.append(minZeroLoc)
        print i
        print zeroSet
        self.zeroSet=zeroSet
        if i==self.m:
            return True
        return False

    '''画零盖线函数'''
    def coverZeroLine(self):
        rowFlag=[True for i in range(self.m)]
        colFlag=[False for i in range(self.n)]
        zeroLocCR={self.zeroSet[i][1]:self.zeroSet[i][0] for i in range(len(self.zeroSet))}
        #实现对没有独立零元素的行打勾  
        for i in range(len(self.zeroSet)):
            rowFlag[self.zeroSet[i][0]]=False
        flag=True#一次大循环中是否有划线标志   
        while flag:
            #对打勾的行所含0元素的列打勾
            flag=False
            for i in range(self.m):
                if not rowFlag[i]:
                    continue
                for j in range(self.n):
                    if self.matr[i,j]==0 and not colFlag[j]:
                        colFlag[j]=True
                        flag=True
            #对所有打勾的列中所含独立0元素的行打勾
            for j in range(self.n):
                if colFlag[j] and zeroLocCR.has_key(j) and not rowFlag[zeroLocCR[j]]:
                    rowFlag[zeroLocCR[j]]=True
                    flag=True
        self.rowFlag=rowFlag
        self.colFlag=colFlag
        print rowFlag
        print colFlag
        
    '''矩阵更新函数'''    
    def updateMatr(self):
        Min=numpy.inf#在零盖线外的元素的最小值，即满足rowFlag[i]=true且colFlag[j]=false的matr[i,j]这些元素中的最小值
        #找到这个最小值
        for i in range(self.m):
            if self.rowFlag[i]:
                for j in range(self.n):
                    if not self.colFlag[j] and self.matr[i,j]<Min:
                        Min=self.matr[i,j]
        #对所有零盖线外的元素减去这个最小值,对被两条零盖线划到的元素加上这个最小值
        for i in range(self.m):
            if self.rowFlag[i]:
                for j in range(self.n):
                    if not self.colFlag[j]:
                        self.matr[i,j]-=Min
            else:
                for j in range(self.n):
                    if  self.colFlag[j]:
                        self.matr[i,j]+=Min
    
    '''总指派函数'''
    def hungary(self):
        self.reduce()
        while not self.tryMatch():
            self.coverZeroLine()
            self.updateMatr()
        matchMatr=numpy.zeros((self.m,self.n))
        for i in range(len(self.zeroSet)):
            matchMatr[self.zeroSet[i][0],self.zeroSet[i][1]]=1
        return matchMatr
        
matr=numpy.mat([[12,7,9,7,9],[8,9,6,6,6],[7,17,12,14,9],[15,14,6,6,10],[4,10,7,10,9]])
#matr=numpy.mat([[1 for j in range(5)] for i in range(5)])
print matr
hungary=Hungary(matr)
'''
hungary.reduce()
print hungary.getMatr()[0,0]==1
if hungary.tryMatch():
    print "匹配成功"
else:
    print "匹配失败"
hungary.coverZeroLine()
hungary.updateMatr()
'''
print hungary.hungary()
print hungary.getMatr()