# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 10:10:35 2020

@author: wh
"""
#第一个程序
#a = 1
#b = 2
#sum = a + b
#print("sum is:{}".format(sum))
#print("sum is:{}".format(a+b))


import numpy as np

a = np.zeros((2,3))
print("2行3列的0矩阵：".format(a))
[m,n]=a.shape
print("0矩阵的行m为{}，列n为{}".format(m,n))

for i in range(0,m):
    print(i)
    
