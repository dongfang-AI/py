# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 22:22:08 2020

@author: wh
"""
#引入包
import numpy as np
#import matplotlib.pyplot as plt
import gym

# 常量设定
ENV = 'CartPole-v0'  # 任务名称
GAMMA = 0.99  # 时间折扣率
MAX_STEPS = 200  # 1次试验中步数
NUM_EPISODES = 1000  # 最大尝试次数

NUM_PROCESSES = 32  # 同时执行环境
NUM_ADVANCED_STEP = 5  # 设置提前计算奖励总和的步数

# A2C的误差函数的常量设置
value_loss_coef = 0.5 #价值损失系数
entropy_coef = 0.01 #信息熵系数
max_grad_norm = 0.5

#==================================================
#存储类定义
#==================================================
class RolloutStorage(object):
    '''Advantage学习的存储类'''

    def __init__(self, num_steps, num_processes, obs_shape):

        self.observations = torch.zeros(num_steps + 1, num_processes, 4)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, 1).long()

        # 存储折扣奖励总和
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.index = 0  # 要insert的索引

    def insert(self, current_obs, action, reward, mask):
        '''存储transition到下一个index'''
        self.observations[self.index + 1].copy_(current_obs)
        self.masks[self.index + 1].copy_(mask)
        self.rewards[self.index].copy_(reward)
        self.actions[self.index].copy_(action)

        self.index = (self.index + 1) % NUM_ADVANCED_STEP  # 更新索引

    def after_update(self):
        '''当Advantage的step数已经完成时，最新的一个存储在index0'''
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])+98


    def compute_returns(self, next_value):
        '''计算Advantage步骤中每个步骤的折扣奖励总和'''

        # 注意： 从5 step后，开始反向计算
        # 注意：第 5 step 是 Advantage1，第4 步 是Advantage2
        self.returns[-1] = next_value
        for ad_step in reversed(range(self.rewards.size(0))):
            self.returns[ad_step] = self.returns[ad_step + 1] * \
                GAMMA * self.masks[ad_step + 1] + self.rewards[ad_step]

#==================================================
#深度神经网络
#==================================================
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.actor = nn.Linear(n_mid, n_out)  # 因为动作已决定，输出就是动作类型的数量
        self.critic = nn.Linear(n_mid, 1)  # 因为他是一个状态价值，输出1

    def forward(self, x):
        '''定义网络前向计算'''
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        critic_output = self.critic(h2)  # 状态价值计算
        actor_output = self.actor(h2)  # 动作的计算

        return critic_output, actor_output

    def act(self, x):
        '''按照概率求状态x的动作'''
        value, actor_output = self(x)
        # dim=1，在动作类型方向计算softmax
        action_probs = F.softmax(actor_output, dim=1)
        action = action_probs.multinomial(num_samples=1)  # dim=1的动作类型方向的概率计算
        return action

    def get_value(self, x):
        '''从状态x获得状态价值'''
        value, actor_output = self(x)

        return value

    def evaluate_actions(self, x, actions):
        '''从状态x获取状态值，记录实际动作的对数概率和熵'''
        value, actor_output = self(x)

        log_probs = F.log_softmax(actor_output, dim=1)  # 使用dim=1在动作类型方向上计算
        action_log_probs = log_probs.gather(1, actions)  # 求实际动作的log_probs

        probs = F.softmax(actor_output, dim=1)  # 在dim=1的动作类型方向上计算
        entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, entropy

#==================================================
#定义agent的大脑类并在所有agent之间共享它们
#==================================================
import torch
from torch import optim

class Brain(object):
    def __init__(self, actor_critic):
        self.actor_critic = actor_critic  # actor_critic是一个Net类深度神经网络
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.01)

    def update(self, rollouts):
        '''对使用Advantage计算所有5个步骤进行更新'''
        obs_shape = rollouts.observations.size()[2:]  # torch.Size([4, 84, 84])
        num_steps = NUM_ADVANCED_STEP
        num_processes = NUM_PROCESSES

        values, action_log_probs, entropy = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1, 4),
            rollouts.actions.view(-1, 1))

        # 注意：每个变量的大小
        # rollouts.observations[:-1].view(-1, 4) torch.Size([80, 4])
        # rollouts.actions.view(-1, 1) torch.Size([80, 1])
        # values torch.Size([80, 1])
        # action_log_probs torch.Size([80, 1])
        # entropy torch.Size([])

        values = values.view(num_steps, num_processes,1)  # torch.Size([5, 16, 1])
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        # advantage的计算（动作价值-状态价值）
        advantages = rollouts.returns[:-1] - values  # torch.Size([5, 16, 1])

        # Critic的损失loss的计算
        value_loss = advantages.pow(2).mean()

        # 计算Actor的gain，然后添加负号以使其作为loss
        action_gain = (action_log_probs*advantages.detach()).mean()
        # detach并将advantages视为常数

        # 误差函数的总和
        total_loss = (value_loss * value_loss_coef -
                      action_gain - entropy * entropy_coef)

        # 更新连接参数
        self.actor_critic.train()  # 在训练模式中
        self.optimizer.zero_grad()  # 重置梯度
        total_loss.backward()  # 计算反向传播
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm)
        #  使梯度大小最大为0.5，以便连接参数不会一下子改变太多

        self.optimizer.step()  # 更新连接参数


#import copy
class Environment:
    def run(self):
        '''主要运行'''

        # 为要同时执行的环境数生成envs
        envs = [gym.make(ENV) for i in range(NUM_PROCESSES)]

        # 生成所有agent共享的脑Brain
        n_in = envs[0].observation_space.shape[0]  # 状态数量为4
        n_out = envs[0].action_space.n  # 动作数量为2
        n_mid = 32
        actor_critic = Net(n_in, n_mid, n_out)  # 生成深度神经网络
        global_brain = Brain(actor_critic)

        # 存储变量生成
        obs_shape = n_in
        current_obs = torch.zeros(NUM_PROCESSES, obs_shape)  # torch.Size([16, 4])
        rollouts = RolloutStorage(NUM_ADVANCED_STEP, NUM_PROCESSES, obs_shape)  # rollouts对象
        episode_rewards = torch.zeros([NUM_PROCESSES, 1])  # 保存当前实验的奖励
        final_rewards = torch.zeros([NUM_PROCESSES, 1])  # 保存最后实验的奖励
        obs_np = np.zeros([NUM_PROCESSES, obs_shape])  # Numpy数组
        reward_np = np.zeros([NUM_PROCESSES, 1])  # Numpy数组
        done_np = np.zeros([NUM_PROCESSES, 1])  # Numpy数组
        each_step = np.zeros(NUM_PROCESSES)  # 记录每个环境中的step数
        episode = 0  # 环境0的实验

        # 初始状态
        obs = [envs[i].reset() for i in range(NUM_PROCESSES)]
        obs = np.array(obs)
        obs = torch.from_numpy(obs).float()  # torch.Size([16, 4])
        current_obs = obs  # 存储最新的obs

        # 将当前状态保存到对象rollouts的第一个状态以进行advanced学习
        rollouts.observations[0].copy_(current_obs)

        # 运行循环
        for j in range(NUM_EPISODES*NUM_PROCESSES):  # for循环整体代码
            # 计算advanced学习的每个step数
            for step in range(NUM_ADVANCED_STEP):

                # 求取行动
                with torch.no_grad():
                    action = actor_critic.act(rollouts.observations[step])

                # (16,1)→(16,)→tensor到NumPy
                actions = action.squeeze(1).numpy()

                # 运行1 step
                for i in range(NUM_PROCESSES):
                    obs_np[i], reward_np[i], done_np[i], _ = envs[i].step(actions[i])

                    # 判断当前episode是否终止以及是否有state_next
                    if done_np[i]:  # 如果步数已超过200，或者杆倾斜超过某个角度，done为true

                        # 仅在环境0时输出
                        if i == 0:
                            print('%d Episode: Finished after %d steps' % (
                                episode, each_step[i]+1))
                            episode += 1

                        # 设置奖励
                        if each_step[i] < 195:
                            reward_np[i] = -1.0  # 如果中途倒下则奖励-1
                        else:
                            reward_np[i] = 1.0  # 站立到结束时奖励1

                        each_step[i] = 0  # step数重置
                        obs_np[i] = envs[i].reset()  # 重复执行环境

                    else:
                        reward_np[i] = 0.0  # 通常奖励0
                        each_step[i] += 1

                # 将奖励转换为tensor并添加到实验总奖励中
                reward = torch.from_numpy(reward_np).float()
                episode_rewards += reward

                # 对于每个执行环境，如果done，则将mask设置为0，如果继续，则将mask设置为1；
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done_np])

                # 更新最后一次实验的总奖励
                final_rewards *= masks  # 如果正在进行则乘以1并保持原样，否则重置为0
                # 如果完成，乘以0以重置
                final_rewards += (1 - masks) * episode_rewards

                # 更新实验的总次数
                episode_rewards *= masks  # 正在进行的mask是1，所以它保持不变；done时，mask为0；

                # done时，将当前状态设置为全0
                current_obs *= masks

                # current_obs更新
                obs = torch.from_numpy(obs_np).float()  # torch.Size([16, 4])
                current_obs = obs  # 存储最新的obs

                # 现在将step的transition放入储存对象
                rollouts.insert(current_obs, action.data, reward, masks)

            # 结束advanced的for

            # 从advanced的最终step的状态计算预期的状态价值

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.observations[-1]).detach()
                # rollouts.observations的大小是torch.Size([6, 16, 4])

            # 计算所有步骤的折扣奖励总和并更新rollouts的变量returns
            rollouts.compute_returns(next_value)

            # 网络和rollouts的更新

            global_brain.update(rollouts)
            rollouts.after_update()

            # 如果所有NUM_PROCESSES都连续超过200 step，则成功
            if final_rewards.sum().numpy() >= NUM_PROCESSES:
                print('连续成功')
                break

# main学学习
cartpole_env = Environment()
cartpole_env.run()














