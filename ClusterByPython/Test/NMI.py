#coding=utf-8
#原理参考:http://www.cnblogs.com/ziqiao/archive/2011/12/13/2286273.html
import numpy

class NMI:
    def __init__(self,clusterAssment=[{}],realAssment=[{}]):
        self.clusterAssment=clusterAssment
        self.realAssment=realAssment
        #self.base=base
    
    '''计算类分布概率函数'''
    def ProbV(self):
        #获取数据点总数并保存
        self.dataNum=sum([len(self.realAssment[i]) for i in range(len(self.realAssment))])
        #计算cluster聚类分布概率并保存
        self.clusterProbV=[float(len(self.clusterAssment[i]))/self.dataNum for i in range(len(self.clusterAssment))]
        #计算真实类别分布概率并保存
        self.realProbV=[float(len(self.realAssment[i]))/self.dataNum for i in range(len(self.realAssment))]
        
    '''计算联合分布概率函数'''
    def joinProb(self):
        self.joinP=numpy.zeros((len(self.clusterAssment),len(self.realAssment)))
        for i in range(len(self.clusterAssment)):
            for j in range(len(self.realAssment)):
                self.joinP[i,j]=float( len(self.clusterAssment[i]&self.realAssment[j]) )/self.dataNum

    '''MI计算函数'''
    def MI(self):
        return numpy.sum(numpy.mat([[self.joinP[i,j]*numpy.log2(self.joinP[i,j]/self.clusterProbV[i]/self.realProbV[j]) for j in range(len(self.realAssment))] for i in range(len(self.clusterAssment))]))
        
    '''熵计算函数'''
    def entropy(self):
        self.clusterEntropy=sum([-self.clusterProbV[i]*numpy.log2(self.clusterProbV[i]) for i in range(len(self.clusterProbV))])
        self.realEntropy=sum([-self.realProbV[i]*numpy.log2(self.realProbV[i]) for i in range(len(self.realProbV))])
    
    '''NMI计算函数'''
    def nmi(self):
        self.ProbV()
        self.joinProb()
        self.entropy()
        return 2*self.MI()/(self.clusterEntropy+self.realEntropy)

'''    
cluster=[{1,2,3,4,6},{6,7,8,9,5}]
real=[{1,2,3,4,5},{6,7,8,9,10}]
nmi=NMI(cluster,real)
print nmi.nmi()
'''