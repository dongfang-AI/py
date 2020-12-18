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
#2.策略参数theta
#==================================================
#将策略参数theta0，转换为随机策略
def simple_convert_into_pi_from_theta(theta):
    '''简单计算比率'''

    [m, n] = theta.shape  # theta矩阵大小
    pi = np.zeros((m, n))
    for i in range(0, m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])  # 计算比率

    pi = np.nan_to_num(pi)  # 将nan转换为0

    return pi

# 求取随机行动策略pi_0
pi_0 = simple_convert_into_pi_from_theta(theta_0)
#打印查看策略pi
print(pi_0)

# 设置初始的动作价值函数
[a, b] = theta_0.shape  # 将行列数放入a, b
Q = np.random.rand(a, b) * theta_0 * 0.1
#将theta_0乘到各元素上，使得Q的墙壁方向的值为nan

#==================================================
#3.动作函数，状态获取函数
#==================================================
# ε-greedy贪婪法
def get_action(s, Q, epsilon, pi_0):
    direction = ["up", "right", "down", "left"]

    # 确定行动
    if np.random.rand() < epsilon:
        # ε概率随机行动
        next_direction = np.random.choice(direction, p=pi_0[s, :])
    else:
        # 采用Q的最大值对应的动作
        next_direction = direction[np.nanargmax(Q[s, :])]

    # 为动作加索引
    if next_direction == "up":
        action = 0
    elif next_direction == "right":
        action = 1
    elif next_direction == "down":
        action = 2
    elif next_direction == "left":
        action = 3

    return action

def get_s_next(s, a, Q, epsilon, pi_0):
    direction = ["up", "right", "down", "left"]
    next_direction = direction[a]  # 行动作a对应的方向

    # 由动作确定下一个状态
    if next_direction == "up":
        s_next = s - 3  # 向上移动，状态-3
    elif next_direction == "right":
        s_next = s + 1  # 向右移动，状态+1
    elif next_direction == "down":
        s_next = s + 3  # 向下移动，状态+3
    elif next_direction == "left":
        s_next = s - 1  # 向左移动，状态-1

    return s_next

#==================================================
#4.Q-learning 程序
#==================================================
#基于Q学习更新动作价值函数Q
def Q_learning(s, a, r, s_next, Q, eta, gamma):

    if s_next == 8:  # 到达目标时
        Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
    else:
        Q[s, a] = Q[s, a] + eta * (r + gamma * np.nanmax(Q[s_next,: ]) - Q[s, a])

    return Q

#==================================================
#5.Q-learning 主函数
#==================================================
#定义基于sarsa求解迷宫问题的函数，输出状态，动作的历史记录以及更新后的Q
def goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi):
    s = 0  # 开始地点
    a = a_next = get_action(s, Q, epsilon, pi)  # 初始动作
    s_a_history = [[0, np.nan]]  # 记录智能体的移动序列

    while (1):  # 循环直至到达目标
        a = a_next  # 跟新动作
        s_a_history[-1][1] = a
        # 将动作放在现在的状态，最终的index=-1
        s_next = get_s_next(s, a, Q, epsilon, pi)
        # 有效的下一个状态
        s_a_history.append([s_next, np.nan])
        # 代入下一个状态，动作未知时为nan
        # 给予奖励，求得下一个动作
        if s_next == 8:
            r = 1  # 到达目标，给予奖励
            a_next = np.nan
        else:
            r = 0
            a_next = get_action(s_next, Q, epsilon, pi)
            # 求得下一动作a_next
        # 更新价值函数
        Q = Q_learning(s, a, r, s_next, Q, eta, gamma)
        # 终止判断
        if s_next == 8:  # 到达目的地结束
            break
        else:
            s = s_next

    return [s_a_history, Q]


#通过Q学习求解迷宫问题
eta = 0.1  # 学习率
gamma = 0.9  # 时间折扣率
epsilon = 0.5  # ε-greedy算法的初始值
v = np.nanmax(Q, axis=1)  # 根据状态求价值的最大
is_continue = True
episode = 1

V=[] #存放每回合的状态价值
V.append(np.nanmax(Q,axis=1)) #求各状态下动作价值的最大值

while is_continue:  # 循环直到is_continue为false
    print("当前回合:" + str(episode))

    # ε-greedy贪婪法的值逐渐减少
    epsilon = epsilon / 2

    # Sarsa求解迷宫问题，求取移动历史和更新后的Q值
    [s_a_history, Q] = goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi_0)

    # 状态价值的变化
    new_v = np.nanmax(Q, axis=1)  # 各状态求得最大价值
    print(np.sum(np.abs(new_v - v)))  # 输出状态价值的变化
    v = new_v
    V.append(v) #添加该回合终止时的状态价值函数

    print("求解迷宫问题所需步数" + str(len(s_a_history) - 1) + "步")

    # 100回合训练
    episode = episode + 1
    if episode > 100:
        break



#将智能体移动进行可视化
from matplotlib import animation
#from IPython.display import HTML
import matplotlib.cm as cm #color map

def init():
    '''初始化背景'''
    line.set_data([], [])
    return (line,)


def animate(i):
    #各帧的绘图内容
    #各方格中根据状态价值的大小画颜色
    line, = ax.plot([0.5], [2.5], marker="s",
                    color=cm.jet(V[i][0]), markersize=85)  # S0
    line, = ax.plot([1.5], [2.5], marker="s",
                    color=cm.jet(V[i][1]), markersize=85)  # S1
    line, = ax.plot([2.5], [2.5], marker="s",
                    color=cm.jet(V[i][2]), markersize=85)  # S2
    line, = ax.plot([0.5], [1.5], marker="s",
                    color=cm.jet(V[i][3]), markersize=85)  # S3
    line, = ax.plot([1.5], [1.5], marker="s",
                    color=cm.jet(V[i][4]), markersize=85)  # S4
    line, = ax.plot([2.5], [1.5], marker="s",
                    color=cm.jet(V[i][5]), markersize=85)  # S5
    line, = ax.plot([0.5], [0.5], marker="s",
                    color=cm.jet(V[i][6]), markersize=85)  # S6
    line, = ax.plot([1.5], [0.5], marker="s",
                    color=cm.jet(V[i][7]), markersize=85)  # S7
    line, = ax.plot([2.5], [0.5], marker="s",
                    color=cm.jet(1.0), markersize=85)  # S8
    return (line,)


#用初始化函数和绘图函数来生成动画
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(V), 
                               interval=200, repeat=False)

#HTML(anim.to_jshtml()) #在jupyter中显示动画























