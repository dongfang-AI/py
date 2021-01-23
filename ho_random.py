# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 17:16:31 2020
功能：随机策略进行切换管理
@author: wh
"""
#导入包
import numpy as np
import matplotlib.pyplot as plt
import math

#######################################################
#1、UE移动轨迹采样
#######################################################
#基站BS部署位置(x,y)，共31个基站，1颗卫星
pos_BS=[[0.6,1.0],[0.6,2.5],[0.6,4],[0.6,5.5],[0.6,7],
        [1.9,1.2],[1.9,2.5],[1.9,4],[1.9,5.5],[1.9,7],
        [3.5,1.4],[3.5,2.7],[3.5,4.5],[3.5,6],[3.5,7],
        [5.1,1.4],[5.0,3],[5.1,4.5],[5.1,6],[5.1,7],
        [6.5,1.0],[6.5,3],[6.5,4.5],[6.5,6],[6.5,7],
        [8.0,4.0],[8,5.5],[8,7],[9.5,4.0],[9.5,5.5],
        [9.5,7]
        ]
node_num = 100 #移动UE数为100
time_len = 600 #采样时长为600s，每隔0.1s采样一次
q = [[],[]] #animation使用，记录移动UE选择路线
speed=[] #animation使用，记录移动UE的速度

#random_ue():产生UE的移动轨迹
#移动UE：随机初始位置，固定方向，随机速度
def random_ue():
    #UE的随机初始位置，移动速度
    global q,speed
    path_list=[0,1,2,3,4,5,6,7,8,9,10]  #11条可选路径的编号列表
    path_num=np.random.choice(path_list) #随机选择的路径编号，path为路径列表
    path = [[[3,8],[3,3.5]],[[1,8],[1,0]],[[0,6.5],[10,6.5]],
            [[0,5],[10,5]],[[0,3.5],[10,3.5]],[[0,2],[7,2]],
            [[1,0],[1,8]],[[7,0],[7,6.5]],[[10,3.5],[0,3.5]],
            [[10,5],[0,5]],[[10,6.5],[0,6.5]]
            ]
    print(path_num,path[path_num]) #打印选择的路径编号和路径位置
    q[0].append(path[path_num][0]) #路径的起始坐标
    q[1].append(path[path_num][1]) #路径的结束坐标
    #dis存储路径中两个坐标的差值
    dis=list(map(lambda x: x[0]-x[1], zip( path[path_num][0],path[path_num][1]))) 
#    print(dis) 
    #随机速度：
    #行人 5km/h 1.4m/s 0.006114 
    #公路 50km/h 14m/s 0.061135 
    #高速 120km/h 33m/s 0.144105 
    ran_v = np.random.choice([0.0006114,0.0061135,0.0144105])
    speed.append(ran_v) #UE速度列表，元素添加
    #根据路径坐标差值dis，判断UE运动方向和运动范围
    x=y=0 #存储UE选择路径的不变轴
    if abs(dis[0]) > abs(dis[1]): #x坐标在变
        if dis[0] > 0: #x坐标递减
            v = -ran_v #坐标值减小，所以是负的速度
            begin = path[path_num][0][0] #路径的开始x
            end = path[path_num][1][0] #路径的结束x
            y = path[path_num][0][1] #路径的y
        else: #x坐标递增
            v = ran_v
            begin = path[path_num][0][0]
            end = path[path_num][1][0]
            y = path[path_num][0][1]
    else: #y坐标在变
        if dis[1] > 0: #y坐标递减
            v= -ran_v
            begin = path[path_num][0][1]
            end = path[path_num][1][1]
            x = path[path_num][0][0]
        else: #y坐标递增
            v= ran_v
            begin = path[path_num][0][1]
            end = path[path_num][1][1]
            x = path[path_num][0][0]
        
#    print(begin,end,x,y,v)
    #调用函数，产生随机运动UE的路径和速度
    #begin,end,x,y,v =  random_ue()
    
    #UE随机移动的坐标采样
    range_value=np.arange(begin,end,v)
    #生成一个0矩阵，存储UE每0.1s的采样坐标
    xy_ue =np.zeros((len(range_value), 2))
    #更新UE位置坐标
    if x > y:
        #[[3,8],[3,3.5]]路径中，x轴不变，x=3,y=0
    #    print('x')
        for i in range(len(range_value)):
            xy_ue[i][0] = x
            xy_ue[i][1] = range_value[i]
    else:
        #y轴不变
    #    print('y')
        for i in range(len(range_value)):
    #        print(range_value[i])
            xy_ue[i][0] = range_value[i]
            xy_ue[i][1] = y
    return xy_ue


#######################################################
#2、计算UE的连接矩阵
#######################################################
#ue_pos位置矩阵:100*12000
ue_pos = np.full((node_num,time_len*2*10), np.nan)
for i in range(node_num):
    print('ue_pos: {}'.format(i))
    k=0
    #生成一个移动UE对应的移动轨迹坐标
    xy_ue = random_ue()
    for j in range(len(xy_ue)):
        ue_pos[i][k]=xy_ue[j][0]
        ue_pos[i][k+1]=xy_ue[j][1]
        k=k+2
        if k+1 > time_len*2*10:
            break
        
#ue_dis距离矩阵:100*6000*31
ue_dis = np.full((node_num,time_len*10,len(pos_BS)), np.nan)
i=j=k=0 #计数变量初始化
for i in range(node_num): #节点数
    print('ue_dis: {}'.format(i))
    m = 0 #ue_pos出现nan，表明节点运动到终点，结束距离计算
    for j in range(time_len*10): #采样数
        if np.isnan(ue_pos[i][m]):
#            print(j) #结束采样编号
            break
        for k in range(len(pos_BS)): #基站数
            ue_x = ue_pos[i][m]
            ue_y = ue_pos[i][m+1]
            bs_x = pos_BS[k][0]
            bs_y = pos_BS[k][1]
            ue_dis[i][j][k]=math.sqrt((ue_x-bs_x)**2+(ue_y-bs_y)**2)
        m=m+2 #一次遍历ue_pos[i]的2个坐标

#ue_bs连接矩阵：100*6000
ue_bs = np.full((node_num,time_len*10), np.nan)
i=j=0 #计数变量初始化
for i in range(node_num):
    print('ue_bs: {}'.format(i))
    for j in range(time_len*10):
        if np.isnan(ue_dis[i][j][0]):
            break
        else:
            ue_bs[i][j] = np.where(ue_dis[i][j]==np.min(ue_dis[i][j]))[0][0]

#######################################################
#3、UE切换统计
#######################################################
#ue切换统计
ue_ho = np.full((node_num,time_len*10), np.nan)
i=j=0 #计数变量初始化
for i in range(ue_bs.shape[0]):
    print("node:{}".format(i))
    for j in range(ue_bs.shape[1]):
        if np.isnan(ue_bs[i][j]) or j ==ue_bs.shape[1]-1 :
            break
        else:
            if ue_bs[i][j] == ue_bs[i][j+1]:
                ue_ho[i][j] = 0
            else:
                ue_ho[i][j] = 1

#每个节点HO次数
ue_ho_num = np.full((node_num,1), np.nan)
for i in range(ue_ho.shape[0]):
    ue_ho_num[i] = np.nansum(ue_ho[i])

#每个采样点HO次数
sample_ho_num = np.full((1,time_len*10), np.nan)
for j in range(ue_ho.shape[1]):
    sample_ho_num[0][j] = np.nansum(ue_ho[:,j])











#######################################################
#移动UE动态显示
#######################################################
#将智能体移动进行可视化
from matplotlib import animation

fig = plt.figure(figsize=(10, 8))
ax = plt.gca()
#设定画图的范围
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
plt.tick_params(axis='both', which='both', bottom='off', top='off',
                labelbottom='off', right='off', left='off', labelleft='off')
#画出道路结构：4横3纵
plt.plot([1, 1], [0, 8], color='lightgrey', linewidth=20)
plt.plot([3, 3], [3.5, 8], color='lightgrey', linewidth=20)
plt.plot([7, 7], [0, 6.5], color='lightgrey', linewidth=20)

plt.plot([0, 10], [6.5, 6.5], color='lightgrey', linewidth=20)
plt.plot([0, 10], [5, 5], color='lightgrey', linewidth=20)
plt.plot([0, 10], [3.5, 3.5], color='lightgrey', linewidth=20)
plt.plot([0, 7], [2, 2], color='lightgrey', linewidth=20)

#设置基站位置
def plt_BS(a,b,r):
    #a为BS的x坐标，b为BS的y坐标，r为BS覆盖范围半径
    plt.plot([a],[b], marker="^", color='skyblue', markersize=20)
    plt.text(a-0.06,b-0.15,'BS',color='black')
    theta = np.arange(0, 2*np.pi, 0.01)
    x = a + r * np.cos(theta)
    y = b + r * np.sin(theta)
    plt.plot(x, y,color="b",linewidth=1.0, linestyle='--')

r=1.1 #基站通信覆盖半径
for a,b in pos_BS:
    plt_BS(a,b,r)

#LEO卫星
plt.plot(8.5,2,marker="*", color='skyblue', markersize=30)
plt.text(8.40,1.85,'LEO',color='black')

#ue采样的位置矩阵
ue_x = np.full((node_num,time_len*10), 10.0)
ue_y = np.full((node_num,time_len*10), 10.0)
#UE的采样坐标
for i in range(node_num):
    k=0
    for j in range(0,len(ue_pos[0]),2):
        if np.isnan(ue_pos[i][j]) or np.isnan(ue_pos[i][j+1]):
            break
        else:
            ue_x[i][k] = ue_pos[i][j]
            ue_y[i][k] = ue_pos[i][j+1]
            k=k+1

x = ue_x[:,0]
y = ue_y[:,0]
#画出初始UE节点位置
points, = ax.plot(x, y,'or',markersize=10)
txt = plt.title('')

#链路显示：方法1 （直线法） 31帧后卡顿
##ue的连接矩阵
#ue_link = [] 
##0-6000 个采样点
#for i in range(time_len*10):
#    ue_link.append([]) #初始化6000*1的空列表
#    #100个节点
#    for j in range(node_num):
#        if np.isnan(ue_bs[j][i]):
#            break
#        else:
#            #ue-BS的x坐标
#            ue_link[i].append([ue_x[j][i],pos_BS[int(ue_bs[j][i])][0]])
#            #ue-BS的y坐标
#            ue_link[i].append([ue_y[j][i],pos_BS[int(ue_bs[j][i])][1]])
#
#lines = ax.plot(*ue_link[0], color='r')
##print(*ue_link[0])

#链路显示：方法2 （20点法）
#ue移动过程中link表示
link_node = 20
link_x = np.full((node_num*link_node,time_len*10), 10.0)
link_y = np.full((node_num*link_node,time_len*10), 10.0)

k=0 #节点编号 100
#0-2000 20步长
for i in range(0,node_num*link_node,20):
    print('link_node:{}'.format(k))
    m=0 #采样编号 6000
    #0-12000 步长为2 
    for j in range(0,time_len*10*2,2):
#        print(j)
        #ue的位置坐标
        if np.isnan(ue_pos[k][j]):
            break
        else:
            x1 = ue_pos[k][j]
            y1 = ue_pos[k][j+1]
            #BS的位置坐标
            x2 = pos_BS[int(ue_bs[k][m])][0]
            y2 = pos_BS[int(ue_bs[k][m])][1]
#            print(x1,y1,x2,y2)
            #初始化ue 0 采样点0 时的链路
            #x坐标
            for n in range(link_node): 
                link_x[i+n][m] = np.arange(x1,x2,-(x1-x2)/link_node)[n]
            #y坐标
            for n in range(link_node):
                link_y[i+n][m] = np.arange(y1,y2,-(y1-y2)/link_node)[n]
            m=m+1
    k=k+1

#画出链路
lines, = ax.plot(link_x[:,0], link_y[:,0],'or', markersize=1)
print(lines)
def update_points(t,ue_x,ue_y,link_x,link_y):  
    txt.set_text('time={:d}'.format(t)) #采样时间
    #node显示
    new_x = ue_x[:,t*10]
    new_y = ue_y[:,t*10]
    print('t:', t)    
    points.set_data(new_x,new_y)
    #link显示
    new_link_x = link_x[:,t*10]
    new_link_y = link_y[:,t*10]
    lines.set_data(new_link_x,new_link_y)
#链路显示方法1-直线法    
#    k=0
#    for i in range(0,len(lines)*2,2):
#        new_link = [ue_link[t*10][i],ue_link[t*10][i+1]]
#        lines[k].set_data(*new_link)
#        k=k+1

    return points,txt,lines

ani=animation.FuncAnimation(fig, update_points, frames=time_len, 
                            fargs=(ue_x,ue_y,link_x,link_y), repeat=False)

#保存htm，html时，同级目录自动存储每帧图像
#ani.save('sin_x.htm')



