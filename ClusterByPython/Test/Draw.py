#coding:utf-8
import numpy as ny
import matplotlib.pyplot as plt

lam=input('请输入λ的值:')
x=ny.arange(0,4)
e=ny.e
y=(1+e**(-1./lam))/(1+e**(x-1./lam))
plt.plot(x,y)
plt.show()