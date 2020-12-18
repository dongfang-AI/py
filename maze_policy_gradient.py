# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 10:55:59 2020
@author: wh
"""
#导入使用的包
import numpy as np
import matplotlib.pyplot as plt
#==================================================
#1.迷宫环境初始化
#==================================================
#迷宫的初始位置
#声明图的大小以及图的变量
fig = plt.figure(figsize=(5, 5))
ax = plt.gca()
#画出红色的墙壁
plt.plot([1, 1], [0, 1], color='red', linewidth=2)
plt.plot([1, 2], [2, 2], color='red', linewidth=2)
plt.plot([2, 2], [2, 1], color='red', linewidth=2)
plt.plot([2, 3], [1, 1], color='red', linewidth=2)
#画出表示状态的文字S0~S8
plt.text(0.5, 2.5, 'S0', size=14, ha='center')
plt.text(1.5, 2.5, 'S1', size=14, ha='center')
plt.text(2.5, 2.5, 'S2', size=14, ha='center')
plt.text(0.5, 1.5, 'S3', size=14, ha='center')
plt.text(1.5, 1.5, 'S4', size=14, ha='center')
plt.text(2.5, 1.5, 'S5', size=14, ha='center')
plt.text(0.5, 0.5, 'S6', size=14, ha='center')
plt.text(1.5, 0.5, 'S7', size=14, ha='center')
plt.text(2.5, 0.5, 'S8', size=14, ha='center')
plt.text(0.5, 2.3, 'START', ha='center')
plt.text(2.5, 0.3, 'GOAL', ha='center')
#设定画图的范围
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)
plt.tick_params(axis='both', which='both', bottom='off', top='off',
                labelbottom='off', right='off', left='off', labelleft='off')
#当前位置S0用绿色圆圈画出
line, = ax.plot([0.5], [2.5], marker="o", color='g', markersize=60)

#==================================================
#2.策略参数theta
#==================================================
#设定参数theta的初始值，用于确定初始方案
#行为状态0~7，列用上↑右→下↓左←表示的移动方向
theta_0 = np.array([[np.nan, 1, 1, np.nan],  # s0
                    [np.nan, 1, np.nan, 1],  # s1
                    [np.nan, np.nan, 1, 1],  # s2
                    [1, 1, 1, np.nan],  # s3
                    [np.nan, np.nan, 1, 1],  # s4
                    [1, np.nan, np.nan, np.nan],  # s5
                    [1, np.nan, np.nan, np.nan],  # s6
                    [1, 1, np.nan, np.nan],  # s7、※s8是目标，无策略
                    ])

#==================================================
#3.策略转换θ->π
#==================================================
#将策略theta根据softmax函数，转换为行动策略pi的函数的定义
def softmax_convert_into_pi_from_theta(theta):
    '''根据softmax函数，计算比率'''
    beta=1.0
    [m, n] = theta.shape  # theta的行和列
    pi = np.zeros((m, n))
    
    exp_theta=np.exp(beta*theta) #将theta转化为exp(theta)
    
    for i in range(0, m):
        pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i, :])  # 计算百分比

    pi = np.nan_to_num(pi)  # 将nan转换为0

    return pi

#求初始策略pi
pi_0 = softmax_convert_into_pi_from_theta(theta_0)
#打印查看策略pi
print(pi_0)

#==================================================
#4.下一状态的获取
#==================================================
#求取动作a,1步移动后求得状态s的函数的定义
def get_action_and_next_s(pi, s):
    direction = ["up", "right", "down", "left"]
    next_direction = np.random.choice(direction, p=pi[s, :])
    # 依据pi[s,:]概率选择direction
    if next_direction == "up":
        action=0
        s_next = s - 3  # 上移动3步
    elif next_direction == "right":
        action=1
        s_next = s + 1  # 右移动1步
    elif next_direction == "down":
        action=2
        s_next = s + 3  # 下移动3步
    elif next_direction == "left":
        action=3
        s_next = s - 1  # 左移动1步

    return [action,s_next]

#==================================================
#5.目标函数
#==================================================
#迷宫内使智能体持续移动的函数
def goal_maze_ret_s_a(pi):
    s = 0  # 开始地点
    s_a_history = [[0,np.nan]]  # 记录智能体移动轨迹的列表
    while (1):  # 死循环，直到到达目标S8
        [action,next_s]=get_action_and_next_s(pi,s)
        s_a_history[-1][1]=action
        #代入当前状态（即目前最后一个状态index=-1）的动作
        s_a_history.append([next_s,np.nan])  # 在记录列表中增加下一个状态（智能体的位置）
        if next_s == 8:  #到达目标地点则终止
            break
        else:
            s = next_s

    return s_a_history

#在迷宫内朝着目标移动
s_a_history = goal_maze_ret_s_a(pi_0)

#打印迷宫随机探索的历史记录
print(s_a_history)
print("迷宫探索步伐：" + str(len(s_a_history) - 1) + "次")

#==================================================
#6.theta的更新
#==================================================
#定义theta的更新函数
def update_theta(theta, pi, s_a_history):
    eta = 0.1 # 学习率
    T = len(s_a_history) - 1  #到达目标的总步数
    [m, n] = theta.shape  # theta矩阵的大小
    delta_theta = theta.copy()  # Δtheta生成、不能直接使用delta_theta = theta

    # 求取delta_theta的各元素
    for i in range(0, m):
        for j in range(0, n):
            if not(np.isnan(theta[i, j])):  # theta不是nan时

                SA_i = [SA for SA in s_a_history if SA[0] == i]
                # 从列表中取出状态i

                SA_ij = [SA for SA in s_a_history if SA == [i, j]]
                # 取出状态i下应该采取的动作

                N_i = len(SA_i)  # 状态i下动作的总次数
                N_ij = len(SA_ij)  # 状态i下采取动作j的次数                
                delta_theta[i, j] = (N_ij - pi[i, j] * N_i) / T

    new_theta = theta + eta * delta_theta

    return new_theta

#策略更新
new_theta = update_theta(theta_0, pi_0, s_a_history)
pi = softmax_convert_into_pi_from_theta(new_theta)
print(pi)

#==================================================
#7.【策略梯度法】求解迷宫问题
#==================================================
stop_epsilon = 10**-4 #策略的变化小于10-4则结束学习
theta = theta_0
pi = pi_0

is_continue = True
count = 1
while is_continue:  # is_continue重复，直到为false
    s_a_history = goal_maze_ret_s_a(pi)  # 由策略pi搜索迷宫探索历史
    new_theta = update_theta(theta, pi, s_a_history)  # 更新参数theta
    new_pi = softmax_convert_into_pi_from_theta(new_theta)  # 更新参数pi

    print(np.sum(np.abs(new_pi - pi)))  # 输出策略的变化
    print("求解迷宫问题所需步数：" + str(len(s_a_history) - 1) + "步")

    if np.sum(np.abs(new_pi - pi)) < stop_epsilon:
        is_continue = False
    else:
        theta = new_theta
        pi = new_pi

np.set_printoptions(precision=3, suppress=True)  # 有効桁数3、指数表示しないという設定
print(pi)


#==================================================
#8.运动轨迹动态显示
#==================================================
#将智能体移动进行可视化
from matplotlib import animation
#from IPython.display import HTML

def init():
    '''初始化背景'''
    line.set_data([], [])
    return (line,)

def animate(i):
    '''每一帧的画面内容'''
    state = s_a_history[i][0]  # 取出s_a_history中每个列表元素中第一个元素，画出当前的位置
    x = (state % 3) + 0.5  # 状态的x坐标为状态数除以3的余数+0.5
    y = 2.5 - int(state / 3)  
    line.set_data(x, y)
    return (line,)


#用初始化函数和绘图函数来生成动画
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(
    s_a_history), interval=200, repeat=False)
























