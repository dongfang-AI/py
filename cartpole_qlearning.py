# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:10:54 2020

@author: wh
"""
#导入包
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import gym

#from JSAnimation.IPython_display import display_animation
#from matplotlib import animation
#from IPython.display import display


#def display_frames_as_gif(frames):
#    """
#    gif形式显示帧的列表，并带有控件
#    """
#    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0),
#               dpi=72)
#    patch = plt.imshow(frames[0])
#    plt.axis('off')
#
#    def animate(i):
#        patch.set_data(frames[i])
#
#    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames),
#                                   interval=50)
#
#    anim.save('movie_cartpole.mp4')  # 保存视频的名字
#    display(display_animation(anim, default_mode='loop'))

#常数的设定
ENV = 'CartPole-v0'  # 使用的任务名称
NUM_DIZITIZED = 6  # 将状态划分为离散值的个数
GAMMA = 0.99  # 时间折扣率
ETA = 0.5  # 学习系数
MAX_STEPS = 200  # 1次试验中的步数
NUM_EPISODES = 1000  # 最大试验次数

class Agent:
    '''CartPole的智能体类，将是带有杆的小车'''

    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)  # 为智能体创建大脑以做出决策

    def update_Q_function(self, observation, action, reward, observation_next):
        '''Q函数的更新'''
        self.brain.update_Q_table(
            observation, action, reward, observation_next)

    def get_action(self, observation, step):
        '''动作的确定'''
        action = self.brain.decide_action(observation, step)
        return action

class Brain:
    '''Agent的大脑，用于Q学习'''

    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  # CartPole两种动作（向左或向右）
        # Q表创建，行数为状态离散化后的分割数，列数为动作数
        self.q_table = np.random.uniform(low=0, high=1, size=(
            NUM_DIZITIZED**num_states, num_actions))
        

    def bins(self, clip_min, clip_max, num):
        '''求得观察到的状态（连续值）到离散值的数字转换阈值'''
        return np.linspace(clip_min, clip_max, num + 1)[1:-1]

    def digitize_state(self, observation):
        '''将观察到的observation转换为离散值'''
        cart_pos, cart_v, pole_angle, pole_v = observation
        digitized = [
            np.digitize(cart_pos, bins=self.bins(-2.4, 2.4, NUM_DIZITIZED)),
            np.digitize(cart_v, bins=self.bins(-3.0, 3.0, NUM_DIZITIZED)),
            np.digitize(pole_angle, bins=self.bins(-0.5, 0.5, NUM_DIZITIZED)),
            np.digitize(pole_v, bins=self.bins(-2.0, 2.0, NUM_DIZITIZED))
        ]
        return sum([x * (NUM_DIZITIZED**i) for i, x in enumerate(digitized)])

    def update_Q_table(self, observation, action, reward, observation_next):
        '''Q学习更新Q表'''
        state = self.digitize_state(observation)  # 状态离散化
        state_next = self.digitize_state(observation_next)  # 将下一个状态离散化
        Max_Q_next = max(self.q_table[state_next][:])
        self.q_table[state, action] = self.q_table[state, action] + \
            ETA * (reward + GAMMA * Max_Q_next - self.q_table[state, action])

    def decide_action(self, observation, episode):
        '''根据ε-greedy贪婪法，逐渐采用最优动作'''
        state = self.digitize_state(observation)
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            action = np.argmax(self.q_table[state][:])
        else:
            action = np.random.choice(self.num_actions)  # 随机返回0,1动作
        return action


class Environment:
    '''CartPole的执行环境'''

    def __init__(self):
        self.env = gym.make(ENV)  # 设置要执行的任务
        num_states = self.env.observation_space.shape[0]  # 获取任务状态个数
        num_actions = self.env.action_space.n  # 获取CartPole的动作数为2
        self.agent = Agent(num_states, num_actions)  # 创建环境中行动的Agent

    def run(self):
        '''执行'''
        complete_episodes = 0  # 195 step以上实验次数
        is_episode_final = False  # 最终实验的标志
#        frames = []  # 存储视频动画的变量

        for episode in range(NUM_EPISODES):  # 实验的最大重复次数
            observation = self.env.reset()  # 环境初始化

            for step in range(MAX_STEPS):  # 每个回合的循环

#                if is_episode_final is True:  # 将最终实验各个时刻的图像添加到帧中
#                    frames.append(self.env.render(mode='rgb_array'))

                # 求取动作
                action = self.agent.get_action(observation, episode)

                # 通过执行动作a_t找到s_{t+1}, r_{t+1}
                observation_next, _, done, _ = self.env.step(
                    action)  # 不使用reward和info
                
                # 给与奖励
                if done:  # 如果步数超过200，或者如果倾斜超过某个角度，则done为true
                    if step < 195:
                        reward = -1  # 如果半途摔倒，给与奖励-1的惩罚
                        complete_episodes = 0  # 站立超过195 step以上，重置实验次数
                    else:
                        reward = 1  # 持续到结束，则奖励为1
                        complete_episodes += 1  # 更新连续记录
                else:
                    reward = 0  # 途中奖励为0

                # 使用step+1的状态observation_next更新Q函数
                self.agent.update_Q_function(
                    observation, action, reward, observation_next)

                # 观测更新
                observation = observation_next

                # 结束时处理
                if done:
                    print('{0} Episode: Finished after {1} time steps'.format(
                        episode, step + 1))
                    break

#            if is_episode_final is True:  # 最后一次实验中保存并绘制动画
#                display_frames_as_gif(frames)
#                break

            if complete_episodes >= 10:  # 10次连续成功，绘制下次实验为最终实验
                print('已经连续成功10次！！')
                break
#                is_episode_final = True  # 最终状态更新


# main
cartpole_env = Environment()
cartpole_env.run()

