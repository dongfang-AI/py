# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 08:24:02 2020

@author: wh
"""
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time


#1.数据预处理
start_time=time.clock()

#mnist数据下载
mnist=fetch_openml('mnist_784',version=1,cache=True)
x= mnist["data"]/255
y=mnist["target"]

stop_time=time.clock()
cost=stop_time - start_time
print("MNIST数据导入耗时：{}\n".format(cost))

#plt.imshow(x[0].reshape(28,28),cmap='gray')
#print("图像数据标签为：{}".format(y[0]))

#2.创建DataLoader
import torch
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split

#2.1将数据分成训练和测试(6:1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/7,random_state=0)

#2.2将数据转化为Pytorch Tensor
x_train=torch.Tensor(x_train)
x_test=torch.Tensor(x_test)
#numpy.object_无法转换为Tensor，需要先强制类型转换
y_train = y_train.astype(float) 
y_test = y_test.astype(float)
y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)

#2.3使用一组数据和标签创建DataSet
ds_train=TensorDataset(x_train,y_train)
ds_test=TensorDataset(x_test,y_test)

#2.4使用小批量数据集创建DataLoader
loader_train=DataLoader(ds_train,batch_size=64,shuffle=True)
loader_test=DataLoader(ds_test,batch_size=64,shuffle=False)


#3.构建神经网络
from torch import nn
#构建的神经网络结构如下：
#输入层：28*28=748个神经元
#中间层：fc1 100个神经元,fc2 100个神经元
#输出层：fc3 10个神经元，对应0~9个数字分类
model=nn.Sequential()
model.add_module('fc1',nn.Linear(28*28*1,100))
model.add_module('relu1',nn.ReLU())
model.add_module('fc2',nn.Linear(100,100))
model.add_module('relu2',nn.ReLU())
model.add_module('fc3',nn.Linear(100,10))

#print(model)

#4.误差函数和优化方法
#误差函数：采用交叉熵作为分类问题的误差函数
#优化方法：采用梯度下降法中的Adam方法
from torch import optim
#误差函数
loss_fn=nn.CrossEntropyLoss() #很多时候选择criterion作为变量名
#优化方法
optimizer=optim.Adam(model.parameters(),lr=0.01)

#5.设置学习和推理
#5.1 学习内容
def train(epoch):
    model.train() #将网络切换到训练模式
    for data,targets in loader_train:
        optimizer.zero_grad() #初始梯度设置为0
        outputs = model(data) #输入数据并计算输出
        loss=loss_fn(outputs,targets) #计算输出和训练数据标签间误差
        loss.backward() #对误差进行反向传播
        optimizer.step()#更新权重
    
    print("epoch{}：结束\n".format(epoch))

#5.2推理内容
def test():
    model.eval() #将网络切换到推理模式
    correct=0
    
    #从数据加载器中取出小批量数据进行计算
    with torch.no_grad(): #输入数据并计算输出
        for data,targets in loader_test:
            outputs=model(data) #找到概率最高的标签
            #推论
            _,predicted=torch.max(outputs.data,1) #找到概率最高的标签
            correct += predicted.eq(targets.data.view_as(predicted)).sum()
            #如果计算结果和标签一致，则计数加1
    
    data_sum=len(loader_test.dataset) #数据的总数
    print("\n准确率:{}/{}({})\n".format(correct,data_sum,100.*correct/data_sum))


#6.执行学习和推理
for epoch in range(3):
    train(epoch)

test()



