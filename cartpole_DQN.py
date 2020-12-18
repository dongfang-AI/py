# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 11:43:49 2020

@author: wh
"""
#导入包
import numpy as np
import matplotlib.pyplot as plt
import gym


#声明动画的绘图函数
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display

def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0),
               dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames),
                                   interval=50)

    anim.save('movie_cartpole_DQN.mp4')  # 動画のファイル名と保存です
    display(display_animation(anim, default_mode='loop'))
    
#实现namedtuple
#使用nametuple与字段名称成对存储值
from collections import namedtuple

Tr = namedtuple('tr', ('name_a', 'value_b'))
Tr_object = Tr('名称为A', 100)

print(Tr_object)  #
print(Tr_object.value_b)  # 输出：100

# namedtuple生成
from collections import namedtuple

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


#常量设定
ENV = 'CartPole-v0'  # 任务名称
GAMMA = 0.99  # 时间折扣率
MAX_STEPS = 200  # 1次试验中的step数
NUM_EPISODES = 500  # 最大尝试次数

#存储经验的内存类
class ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  # memory的最大长度
        self.memory = []  # 存储过往经验
        self.index = 0  # 表示要保存的索引

    def push(self, state, action, state_next, reward):
        '''transition = (state, action, state_next, reward)保存在存储器中'''

        if len(self.memory) < self.capacity:
            self.memory.append(None)  # 内存未满时添加

        # namedtuple对象Transition将值和字段名称保存为一对
        self.memory[self.index] = Transition(state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity  # 保存的index移动1位

    def sample(self, batch_size):
        '''随机检索batch_size大小的样本并返回'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''返回当前memory的长度'''
        return len(self.memory)


#定义Brain类
#将Q函数设置为深度学习网络
        
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

BATCH_SIZE = 32
CAPACITY = 10000


class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  # 获取CartPole的2个动作

        # 创建存储经验的对象
        self.memory = ReplayMemory(CAPACITY)

        # 构建一个神经网络
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, 32))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(32, 32))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, num_actions))

        print(self.model)  # 输出网络的结构

        # 最优化方法的设定
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def replay(self):
        '''Experience Replay学习网络的连接参数'''

        # -----------------------------------------
        # 1. 检查经验池大小
        # -----------------------------------------
        # 1.1 经验池大小
        if len(self.memory) < BATCH_SIZE:
            return

        # -----------------------------------------
        # 2. 创建小批量数据
        # -----------------------------------------
        # 2.1 从经验池中获取小批量数据
        transitions = self.memory.sample(BATCH_SIZE)

        # 2.2 将每个变量转换为与小批量数据对于的形式
        # 得到的transitions存储在BATCH_SIZE(state, action, state_next, reward)
        # 即(state, action, state_next, reward)×BATCH_SIZE
        # 想要把它变成小批量数据
        # 设为(state×BATCH_SIZE, action×BATCH_SIZE, state_next×BATCH_SIZE, reward×BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # 2.3 将每个变量的元素转换为与小批量数据对应的形式
        # 对于state，[torch.FloatTensor of size 1x4]
        # 将其转换为 torch.FloatTensor of size BATCH_SIZEx4 
        # cat是指Concatenates（连接）
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # -----------------------------------------
        # 3. 求取Q(s_t,a_t)值作为监督信号
        # -----------------------------------------
        # 3.1 将网络切换到推理模式
        self.model.eval()

        # 3.2 求取网络输出的Q(s_t,a_t)
        # self.model(state_batch)输出左右两个Q值
        #成为 [torch.FloatTensor of size BATCH_SIZEx2]
        # 为了求得与与此处执行的动作a_t对于的Q值，求取由action_batch执行的动作a_t是向右还是向左的index
        # 用gather获得相应的Q值
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # 3.3 求取max{Q(s_t+1, a)}値，但要注意以下状态

        # 创建索引掩码以检查cartpole是否未完成且具有next_state
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)))
        # 首先全部设置为0
        next_state_values = torch.zeros(BATCH_SIZE)

        # 求取具有下一状态的index的最大Q值
        # 访问输出并通过max(1),求列方向最大值[value,index]
        # そ并输出Q值（index=0）
        # 用detach取出该值
        next_state_values[non_final_mask] = self.model(
            non_final_next_states).max(1)[0].detach()

        # 3.4 从Q(s_t, a_t)求取Q值作为监督信息
        expected_state_action_values = reward_batch + GAMMA * next_state_values

        # -----------------------------------------
        # 4. 连接参数更新
        # -----------------------------------------
        # 4.1 将网络切换到训练模式
        self.model.train()

        # 4.2 计算损失函数（smooth_l1_loss是Huberloss）
        # expected_state_action_values的size是[minbatch]，通过
        # unsqueeze得到[minibatch x 1]
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        # 4.3 更新连接参数
        self.optimizer.zero_grad()  # 重置渐变
        loss.backward()  # 计算反向传播
        self.optimizer.step()  # 更新连接参数

    def decide_action(self, state, episode):
        '''根据当前状态确定动作'''
        # ε-greedy方法逐步采用最佳动作
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()  # 将网络切换到推理模式
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
            # 获取网络输出最大值的索引，index= max(1)[1]
            # .view(1,1)将[torch.LongTensor of size 1]　转换为 size 1x1 形式

        else:
            # 随机返回0,1动作
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]])  # 随机返回0,1动作
            # action的形式为[torch.LongTensor of size 1x1]

        return action

#在CartPole上运行的智能体类，是一个带有杆的小车
class Agent:
    def __init__(self, num_states, num_actions):
        '''设置任务状态和动作数量'''
        self.brain = Brain(num_states, num_actions)  # 为智能体生成大脑来决定他们的动作

    def update_q_function(self):
        '''Q函数更新'''
        self.brain.replay()

    def get_action(self, state, episode):
        '''确定动作'''
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        '''将state, action, state_next, reward的内容保存在经验池中'''
        self.brain.memory.push(state, action, state_next, reward)

#执行CartPole的环境类
class Environment:

    def __init__(self):
        self.env = gym.make(ENV)  # 设定要执行的任务
        num_states = self.env.observation_space.shape[0]  # 设定任务状态和动作的数量
        num_actions = self.env.action_space.n  # CartPole的动作数为2（向左或向右）
        self.agent = Agent(num_states, num_actions)  # 创建Agent在环境中执行的动作

        
    def run(self):
        '''执行'''
        episode_10_list = np.zeros(10)  # 存储10次实验的连续战立步数，用于输出平均步数
        complete_episodes = 0  # 持续站立195步或更多的实验次数
        episode_final = False  # 最终尝试标志
        frames = []  # 用于存储图像的质量，以使最后一轮成为动画

        for episode in range(NUM_EPISODES):  # 环境重复实验次数
            observation = self.env.reset()  # 环境初始化

            state = observation  # 直接使用观测作为状态state使用
            state = torch.from_numpy(state).type(
                torch.FloatTensor)  # 将NumPy变量转换为PyTorch的Tensor
            state = torch.unsqueeze(state, 0)  # size 4转换为size 1x4

            for step in range(MAX_STEPS):  # 1 episode（轮）循环

                if episode_final is True:  # 在最终实验中，将各时刻图像添加到帧中
                    frames.append(self.env.render(mode='rgb_array'))

                action = self.agent.get_action(state, episode)  # 求取动作

                # 通过执行动作a_t求s_{t+1}和done
                # 从action中指定 .item()并获取内容
                observation_next, _, done, _ = self.env.step(
                    action.item())  # 使用"_",是因为reward和info后续流程不适用

                # 给与奖励。对episode是否结束进行判断
                if done:  # step不超过200，或者倾斜超过某个角度，则done为true
                    state_next = None  # 没有下一状态，存储None

                    # 添加到最近的10 episode的站立step列表中
                    episode_10_list = np.hstack(
                        (episode_10_list[1:], step + 1))

                    if step < 195:
                        reward = torch.FloatTensor(
                            [-1.0])  # 半途倒下，奖励-1作为惩罚
                        complete_episodes = 0  # 重置连续成记录
                    else:
                        reward = torch.FloatTensor([1.0])  # 一直站立直到结束时，奖励1
                        complete_episodes = complete_episodes + 1  # 更新连续记录
                else:
                    reward = torch.FloatTensor([0.0])  # 普通奖励为0
                    state_next = observation_next  # 保持观测不变
                    state_next = torch.from_numpy(state_next).type(
                        torch.FloatTensor)  # 将numpy变量转换为PyTorch Tensor
                    state_next = torch.unsqueeze(state_next, 0)  # FloatTensor size 4转换为size 1x4

                # 向经验池中添加经验
                self.agent.memorize(state, action, state_next, reward)

                # Experience Replay中更新Q函数
                self.agent.update_q_function()

                # 更新观测值
                state = state_next

                # 结束处理
                if done:
                    print('%d Episode: Finished after %d steps：10实验平均step数 = %.1lf' % (
                        episode, step + 1, episode_10_list.mean()))
                    break

            if episode_final is True:
                # 保存并绘制动画
                display_frames_as_gif(frames)
                break

            # 连续10轮成功
            if complete_episodes >= 10:
                print('10轮连续成功')
                episode_final = True  # 试下一次尝试成为最终绘制的动画



# main クラス
cartpole_env = Environment()
cartpole_env.run()

