# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 10:41:06 2020

@author: wh
"""
#导入包
import numpy as np
import matplotlib.pyplot as plt
import gym

# namedtuple生成
from collections import namedtuple
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

#常量设定
ENV = 'CartPole-v0'  # 任务名称
GAMMA = 0.99  # 时间折扣率
MAX_STEPS = 200  # 1次试验中的step数
NUM_EPISODES = 500  # 最大尝试次数

#==================================================
#1.经验池类ReplayMemory
#==================================================
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

#==================================================
#2.建立深度学习网络
#==================================================
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output = self.fc3(h2)
        return output


#==================================================
#3.Brain类
#==================================================
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

BATCH_SIZE = 32
CAPACITY = 10000


class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  # CartPole的2个动作

        # 创建经验池
        self.memory = ReplayMemory(CAPACITY)

        # 构建神经网络
        n_in, n_mid, n_out = num_states, 32, num_actions
        self.main_q_network = Net(n_in, n_mid, n_out)  # Net类构造主Q函数
        self.target_q_network = Net(n_in, n_mid, n_out)  # Net类构建目标Q函数
        print(self.main_q_network)  # 输出网络形状

        # 优化方法设定
        self.optimizer = optim.Adam(
            self.main_q_network.parameters(), lr=0.0001)

    def replay(self):
        '''Experience Replay学习网络的连接参数'''

        # 1. 检查经验池的大小
        if len(self.memory) < BATCH_SIZE:
            return

        # 2. 创建小批量数据
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()

        # 3. 找到Q(s_t, a_t)作为监督信息
        self.expected_state_action_values = self.get_expected_state_action_values()

        # 4. 更新参数
        self.update_main_q_network()

    def decide_action(self, state, episode):
        '''根据当前状态确定动作'''
        # 采用ε-greedy方法逐步采用最佳动作
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()  # 将网络切换为推理模式
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1, 1)
            # 获取网络输出最大值的索引index = max(1)[1]
            # .view(1,1)将[torch.LongTensor of size 1]　转换为 size 1x1 

        else:
            # 0,1随机返回
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]])  # 随机返回
            # action的形式为[torch.LongTensor of size 1x1]

        return action

    def make_minibatch(self):
        '''2. 创建小批量数据'''

        # 2.1 从经验池中获取小批量数据
        transitions = self.memory.sample(BATCH_SIZE)

        # 2.2 将每个变量转换为与小批量数据对应的形式
        # transitions表示1 step的(state, action, state_next, reward)对于BATCH_SIZE个
        # 即(state, action, state_next, reward)×BATCH_SIZE
        #它变成小批量数据，即
        # (state×BATCH_SIZE, action×BATCH_SIZE, state_next×BATCH_SIZE, reward×BATCH_SIZE)形式
        batch = Transition(*zip(*transitions))

        # 2.3 将每个变量的元素转换为与小批量数据对应的形式
        # 如，state中[torch.FloatTensor of size 1x4]有BATCH_SIZE个
        # 转换为 torch.FloatTensor of size BATCH_SIZEx4 
        # cat是Concatenates（连接）
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        '''3. 找到Q(s_t,a_t)'''

        # 3.1 将网络切换到推理模式
        self.main_q_network.eval()
        self.target_q_network.eval()

        # 3.2 求网络输出的 Q(s_t, a_t)
        # self.model(state_batch)输出向右或向左的Q值
        # [torch.FloatTensor of size BATCH_SIZEx2]
        # 为了找到与此处执行的动作a_t对应的Q値，找到action_batch执行的动作a_t是向右还是向左的index索引
        # Q値用gather获得
        self.state_action_values = self.main_q_network(
            self.state_batch).gather(1, self.action_batch)

        # 3.3 max{Q(s_t+1, a)}値求得

        # 创建索引掩码，判断cartpole是否未完成且具有next_state
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                    self.batch.next_state)))
        # 首先全部设置为0
        next_state_values = torch.zeros(BATCH_SIZE)

        a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor)

        # 最大Q値的动作a_m从Main Q-Network中求得
        # 最后的[1]返回与该动作对应的索引index
        a_m[non_final_mask] = self.main_q_network(
            self.non_final_next_states).detach().max(1)[1]

        # 仅过滤具有下一个状态的，并将size 32转换为32×1
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # 下一状态的index的动作a_m的Q値从target Q-Network中求得
        # 用detach()取出
        # squeeze()将size[minibatch×1]压缩为[minibatch]。
        next_state_values[non_final_mask] = self.target_q_network(
            self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        # 3.4 根据Q学习公式，求出Q(s_t, a_t)值作为监督信息
        expected_state_action_values = self.reward_batch + GAMMA * next_state_values

        return expected_state_action_values

    def update_main_q_network(self):
        '''4. 更新连接参数'''

        # 4.1 将网络切换到训练模式
        self.main_q_network.train()

        # 4.2 计算损失函数（smooth_l1_loss是Huberloss）
        # expected_state_action_values是
        # size是[minbatch]，unsqueeze到[minibatch x 1]
        loss = F.smooth_l1_loss(self.state_action_values,
                                self.expected_state_action_values.unsqueeze(1))

        # 4.3 更新连接参数
        self.optimizer.zero_grad()  # 重置梯度
        loss.backward()  # 计算反向传播
        self.optimizer.step()  # 更新连接参数

    def update_target_q_network(self):  # DDQN添加
        '''Target Q-Network与主网络相同'''
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

#==================================================
#4.Agent类
#==================================================
class Agent:
    def __init__(self, num_states, num_actions):
        '''设置任务状态和动作的数量'''
        self.brain = Brain(num_states, num_actions)  # 创建一个大脑为Agent决定动作

    def update_q_function(self, episode):
        '''Q函数更新'''
        self.brain.replay(episode)

    def get_action(self, state, episode):
        '''确定动作'''
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        '''将state, action, state_next, reward存储到经验池中'''
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        '''将Target Q-Network更新到与Main Q-Network相同'''
        self.brain.update_target_q_network()

#==================================================
#5.Environment类
#==================================================
class Environment:

    def __init__(self):
        self.env = gym.make(ENV)  # 设置执行的任务
        num_states = self.env.observation_space.shape[0]  # 设置任务状态和动作数量
        num_actions = self.env.action_space.n  # CartPole的2个动作（向左或向右）
        # 创建在环境中行动的Agent
        self.agent = Agent(num_states, num_actions)

    def run(self):
        '''执行'''
        episode_10_list = np.zeros(10)  # 存储10次实验的连续站立步数，输出平均步数
        complete_episodes = 0  # 持续195 step以上实验次数
        episode_final = False  # 最后一轮标志
        frames = []  # 用于存储图像的变量，以使得最后一轮成为动画

        for episode in range(NUM_EPISODES):  # 重复实验次数
            observation = self.env.reset()  # 环境初始化

            state = observation  # 将观测值设为状态值
            state = torch.from_numpy(state).type(
                torch.FloatTensor)  # numpy变量转换为PyTorch Tensor
            state = torch.unsqueeze(state, 0)  # size 4转换为size 1x4

            for step in range(MAX_STEPS):  # 1回合循环
                
                # 将回执动画过程注释掉
                #if episode_final is True:  # 最后一轮中，将各时刻的图像添加到帧中
                    # frames.append(self.env.render(mode='rgb_array'))
                    
                action = self.agent.get_action(state, episode)  # 求要采取的动作

                # 执行动作a_t找到s_{t+1}和done
                # 从action中指定.item()并获取内容
                observation_next, _, done, _ = self.env.step(
                    action.item())  # 不使用reward和info所以设为_

                # 给与奖励，对episode是否结束、state_next是否存在进行判断
                if done:  # 部署超过200，或者如果倾斜超过某个角度，则done为true
                    state_next = None  # 没有下一个状态，因此存储None

                    # 将最近10 episode的站立step数添加到列表
                    episode_10_list = np.hstack(
                        (episode_10_list[1:], step + 1))

                    if step < 195:
                        reward = torch.FloatTensor(
                            [-1.0])  # 如果途中倒下，奖励-1作为惩罚
                        complete_episodes = 0  # 重复连续成功次数
                    else:
                        reward = torch.FloatTensor([1.0])  # 站立直到结束时，给与奖励1
                        complete_episodes = complete_episodes + 1  # 更新连续成功次数
                else:
                    reward = torch.FloatTensor([0.0])  # 通常情况奖励为0
                    state_next = observation_next  # 将状态设置为观测值
                    state_next = torch.from_numpy(state_next).type(
                        torch.FloatTensor)  # numpy变量转换为PyTorch Tensor
                    state_next = torch.unsqueeze(state_next, 0)  # size 4 扩展为size 1x4

                # 为经验池添加经验
                self.agent.memorize(state, action, state_next, reward)

                # Experience Replay更新Q函数
                self.agent.update_q_function()

                # 状态更新
                state = state_next

                # 结束处理
                if done:
                    print('%d Episode: Finished after %d steps：10次试验的平均step数 = %.1lf' % (
                        episode, step + 1, episode_10_list.mean()))
                    
                    # 使用DDQN添加、使Target Q-Network和Main相同
                    if(episode % 2 == 0):
                        self.agent.update_target_q_function()
                    break
                    
                    
            if episode_final is True:
                # 动画注释掉
                # 保存并绘制
                #display_frames_as_gif(frames)
                break

            # 连续成功10轮
            if complete_episodes >= 10:
                print('10轮连续成功')
                episode_final = True  # 使用下一次试验作为最后一轮，从而绘制动画



# main 执行
cartpole_env = Environment()
cartpole_env.run()

