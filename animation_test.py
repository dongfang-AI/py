# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 15:19:19 2020

@author: wh
"""
#############################################################
#1.移动的切线
#############################################################
#import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#
##初始化画布
#fig = plt.figure()
#plt.grid(ls='--')
#
##绘制一条正弦函数曲线
#x = np.linspace(0,2*np.pi,100)
#y = np.sin(x)
#
#crave_ani, = plt.plot(x,y,'red',alpha=0.5) #透明度 0.5
#
##绘制曲线上的切点
#point_ani, = plt.plot(0,0,'r',alpha=0.4,marker='o')
#
##绘制x、y的坐标标识
#xtext_ani = plt.text(5,0.8,'',fontsize=12)
#ytext_ani = plt.text(5,0.7,'',fontsize=12)
#ktext_ani = plt.text(5,0.6,'',fontsize=12)
#
##计算切线的函数
#def tangent_line(x0,y0,k):
#	xs = np.linspace(x0 - 0.5,x0 + 0.5,100)
#	ys = y0 + k * (xs - x0)
#	return xs,ys
#
##计算斜率的函数
#def slope(x0):
#	num_min = np.sin(x0 - 0.05)
#	num_max = np.sin(x0 + 0.05)
#	k = (num_max - num_min) / 0.1
#	return k
#
##绘制切线
#k = slope(x[0])
#xs,ys = tangent_line(x[0],y[0],k)
#tangent_ani, = plt.plot(xs,ys,c='blue',alpha=0.8)
#
##更新函数
#def updata(num):
#	k=slope(x[num])
#	xs,ys = tangent_line(x[num],y[num],k)
#	tangent_ani.set_data(xs,ys)
#	point_ani.set_data(x[num],y[num])
#	xtext_ani.set_text('x=%.3f'%x[num])
#	ytext_ani.set_text('y=%.3f'%y[num])
#	ktext_ani.set_text('k=%.3f'%k)
#	return [point_ani,xtext_ani,ytext_ani,tangent_ani,k]
#
#ani = animation.FuncAnimation(fig=fig,func=updata,frames=np.arange(0,100),interval=100)

#ani.save('sin_x.gif')
#plt.show()

#############################################################
#2.封闭正弦函数图
#############################################################
#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
#
#fig, ax = plt.subplots()   
##生成子图，相当于fig = plt.figure(),ax = fig.add_subplot(),其中ax的函数参数表示把当前画布进行
##分割，例：fig.add_subplot(2,2,2).表示将画布分割为两行两列ax在第2个子图中绘制，其中行优先，
#xdata, ydata = [], []      #初始化两个数组
#ln, = ax.plot([], [], 'r-', animated=False)  #第三个参数表示画曲线的颜色和线型，
#
#def init():
#    ax.set_xlim(0, 2*np.pi)  #设置x轴的范围pi代表3.14...圆周率，
#    ax.set_ylim(-1, 1) #设置y轴的范围
#    return ln,               #返回曲线
#
#def update(n):
#    xdata.append(n)         #将每次传过来的n追加到xdata中
#    ydata.append(np.sin(n))
#    ln.set_data(xdata, ydata)    #重新设置曲线的值
#    return ln,
#
#ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 10),     #这里的frames在调用update函数是会将frames作为实参传递给“n”
#                    init_func=init, blit=True)
##plt.show()

#############################################################
#3.3D多点运动
#############################################################
#from matplotlib import pyplot as plt
#import numpy as np
#import mpl_toolkits.mplot3d.axes3d as p3
#from matplotlib import animation
#
#
#fig = plt.figure()
#ax = p3.Axes3D(fig)
#
#q = [[-4.32, -2.17, -2.25, 4.72, 2.97, 1.74],
#     [ 2.45, 9.73,  7.45,4.01,3.42,  1.80],[-1.40, -1.76, -3.08,-9.94,-3.13,-1.13]]
#v = [[ 0.0068,0.024, -0.014,-0.013, -0.0068,-0.04],[ 0.012,
#      0.056, -0.022,0.016,  0.0045, 0.039],
#     [-0.0045,  0.031,  0.077,0.0016, -0.015,-0.00012]]
#
#x=np.array(q[0])
#y=np.array(q[1])
#z=np.array(q[2])
#s=np.array(v[0])
#u=np.array(v[1])
#w=np.array(v[2])
#
#points, = ax.plot(x, y, z, '*')
#txt = fig.suptitle('')
#
#def update_points(t, x, y, z, points):
#    txt.set_text('num={:d}'.format(t))
#
#    new_x = x + s * t
#    new_y = y + u * t
#    new_z = z + w * t
#    print('t:', t)
#
#    # update properties
#    points.set_data(new_x,new_y)
#    points.set_3d_properties(new_z, 'z')
#
#    # return modified artists
#    return points,txt
#
#ani=animation.FuncAnimation(fig, update_points, frames=15, fargs=(x, y, z, points))
#
#ax.set_xlabel("x [pc]")
#ax.set_ylabel("y [pc]")
#ax.set_zlabel('z [pc]')
#plt.show()


#############################################################
#4.2D多点运动
#############################################################
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import animation


fig = plt.figure()
ax = plt.gca()

#位置
q1 = [[-4.32, -2.17, -2.25, 4.72, 2.97, 1.74],
     [ 2.45, 9.73,  7.45, 4.01, 3.42, 1.80]]
#速度
v1 = [[ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
     [0, 0, 0, 0,  0, 0]]

#链路
#line_x = [[-4.32,0],[ -2.17,0], [-2.25,0], [4.72,0], [2.97,0], [1.74,0]]
#line_y = [[ 2.45,5], [9.73,5],  [7.45,5], [4.01,5], [3.42,5], [1.80,5]]

x=np.array(q1[0])
y=np.array(q1[1])

s=np.array(v1[0])
u=np.array(v1[1])

points, = ax.plot(x, y,  'o')
#ax.plot(line_x,line_y,'r')

#for i in range(len(line_x)):
#    plt.plot(line_x[i],line_y[i],'r')

plt.plot(0,5,marker="*", color='skyblue', markersize=30)
plt.text(-0.2,4.85,'LEO',color='black')

plt.plot(0,4,marker="o", color='skyblue',)

txt = fig.suptitle('')

def update_points(t, x, y, points):    
    txt.set_text('num={:d}'.format(t)) #采样时间

    new_x = x + s * t
    new_y = y + u * t
    print('t:', t)

    # update properties
    points.set_data(new_x,new_y)

    return points,txt

ani=animation.FuncAnimation(fig, update_points, frames=15, fargs=(x, y, points))

ax.set_xlabel("x [pc]")
ax.set_ylabel("y [pc]")

#############################################################
#版本1
#############################################################
##将智能体移动进行可视化
#from matplotlib import animation
#
##移动UE采样时存储的起始位置和随机速度
#ani_q = [[],[]]
#ani_s = [[],[]]
##q:节点的位置
#for i in range(len(q[0])):
#    ani_q[0].append(q[0][i][0])
#    ani_q[1].append(q[0][i][1])
##speed:速度
#for i in range(len(speed)):
#    xv=yv=0
#    if q[0][i][0] != q[1][i][0]:
#        xv = speed[i]
#    else:
#        yv = speed[i]
#    ani_s[0].append(xv)
#    ani_s[1].append(yv)
#
#x=np.array(ani_q[0])
#y=np.array(ani_q[1])
#xs=np.array(ani_s[0])
#ys=np.array(ani_s[1])
#
#points, = ax.plot(x, y,'or')
#txt = fig.suptitle('')
#
#def update_points(t, x, y, points):    
#    txt.set_text('num={:d}'.format(t)) #采样时间
#
#    new_x = x + xs * 10 *t
#    new_y = y + ys * 10 *t
#    print('t:', t)
#
#    # update properties
#    points.set_data(new_x,new_y)
#
#    return points,txt
#
#ani=animation.FuncAnimation(fig, update_points, frames=time_len*10, 
#                            fargs=(x, y, points), repeat=False)



