# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 10:55:59 2020
@author: wh
"""
#导入使用的包
import numpy as np
import matplotlib.pyplot as plt

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

#将策略theta转换为行动策略pi的函数的定义
def simple_convert_into_pi_from_theta(theta):
    '''简单计算百分比'''

    [m, n] = theta.shape  # theta的行和列
    pi = np.zeros((m, n))
    for i in range(0, m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])  # 计算百分比

    pi = np.nan_to_num(pi)  # 将nan转换为0

    return pi

#求初始策略pi
pi_0 = simple_convert_into_pi_from_theta(theta_0)
#打印查看策略pi
print(pi_0)

#1步移动后求得状态s的函数的定义
def get_next_s(pi, s):
    direction = ["up", "right", "down", "left"]

    next_direction = np.random.choice(direction, p=pi[s, :])
    # 依据pi[s,:]概率选择direction

    if next_direction == "up":
        s_next = s - 3  # 上移动3步
    elif next_direction == "right":
        s_next = s + 1  # 右移动1步
    elif next_direction == "down":
        s_next = s + 3  # 下移动3步
    elif next_direction == "left":
        s_next = s - 1  # 左移动1步

    return s_next

#迷宫内使智能体持续移动的函数
def goal_maze(pi):
    s = 0  # 开始地点
    state_history = [0]  # 记录智能体移动轨迹的列表

    while (1):  # 死循环，直到到达目标S8
        next_s = get_next_s(pi, s)
        state_history.append(next_s)  # 在记录列表中增加下一个状态（智能体的位置）

        if next_s == 8:  #到达目标地点则终止
            break
        else:
            s = next_s

    return state_history

#在迷宫内朝着目标移动
state_history = goal_maze(pi_0)

#打印迷宫随机探索的历史记录
print(state_history)
print("迷宫探索步伐：" + str(len(state_history) - 1) + "次")

#将智能体移动进行可视化
from matplotlib import animation
from IPython.display import HTML


def init():
    '''初始化背景'''
    line.set_data([], [])
    return (line,)


def animate(i):
    '''每一帧的画面内容'''
    state = state_history[i]  # 画出当前的位置
    x = (state % 3) + 0.5  # 状态的x坐标为状态数除以3的余数+0.5
    y = 2.5 - int(state / 3)  
    line.set_data(x, y)
    return (line,)


#用初始化函数和绘图函数来生成动画
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(
    state_history), interval=200, repeat=False)

#HTML(anim.to_jshtml()) #在jupyter中显示动画























