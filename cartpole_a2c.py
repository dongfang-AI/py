# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 22:22:08 2020

@author: wh
"""
import numpy as np
import matplotlib.pyplot as plt
import gym

# 定数の設定
ENV = 'CartPole-v0'  # 使用する課題名
GAMMA = 0.99  # 時間割引率
MAX_STEPS = 200  # 1試行のstep数
NUM_EPISODES = 1000  # 最大試行回数

NUM_PROCESSES = 32  # 同時に実行する環境
NUM_ADVANCED_STEP = 5  # 何ステップ進めて報酬和を計算するのか設定


# A2Cの損失関数の計算のための定数設定
value_loss_coef = 0.5
entropy_coef = 0.01
max_grad_norm = 0.5


class RolloutStorage(object):
    '''Advantage学習するためのメモリクラスです'''

    def __init__(self, num_steps, num_processes, obs_shape):

        self.observations = torch.zeros(num_steps + 1, num_processes, 4)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, 1).long()

        # 割引報酬和を格納
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.index = 0  # insertするインデックス

    def insert(self, current_obs, action, reward, mask):
        '''次のindexにtransitionを格納する'''
        self.observations[self.index + 1].copy_(current_obs)
        self.masks[self.index + 1].copy_(mask)
        self.rewards[self.index].copy_(reward)
        self.actions[self.index].copy_(action)

        self.index = (self.index + 1) % NUM_ADVANCED_STEP  # インデックスの更新

    def after_update(self):
        '''Advantageするstep数が完了したら、最新のものをindex0に格納'''
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        '''Advantageするステップ中の各ステップの割引報酬和を計算する'''

        # 注意：5step目から逆向きに計算しています
        # 注意：5step目はAdvantage1となる。4ステップ目はAdvantage2となる。・・・
        self.returns[-1] = next_value
        for ad_step in reversed(range(self.rewards.size(0))):
            self.returns[ad_step] = self.returns[ad_step + 1] * \
                GAMMA * self.masks[ad_step + 1] + self.rewards[ad_step]


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.actor = nn.Linear(n_mid, n_out)  # 行動を決めるので出力は行動の種類数
        self.critic = nn.Linear(n_mid, 1)  # 状態価値なので出力は1つ

    def forward(self, x):
        '''ネットワークのフォワード計算を定義します'''
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        critic_output = self.critic(h2)  # 状態価値の計算
        actor_output = self.actor(h2)  # 行動の計算

        return critic_output, actor_output

    def act(self, x):
        '''状態xから行動を確率的に求めます'''
        value, actor_output = self(x)
        # dim=1で行動の種類方向にsoftmaxを計算
        action_probs = F.softmax(actor_output, dim=1)
        action = action_probs.multinomial(num_samples=1)  # dim=1で行動の種類方向に確率計算
        return action

    def get_value(self, x):
        '''状態xから状態価値を求めます'''
        value, actor_output = self(x)

        return value

    def evaluate_actions(self, x, actions):
        '''状態xから状態価値、実際の行動actionsのlog確率とエントロピーを求めます'''
        value, actor_output = self(x)

        log_probs = F.log_softmax(actor_output, dim=1)  # dim=1で行動の種類方向に計算
        action_log_probs = log_probs.gather(1, actions)  # 実際の行動のlog_probsを求める

        probs = F.softmax(actor_output, dim=1)  # dim=1で行動の種類方向に計算
        entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, entropy


import torch
from torch import optim


class Brain(object):
    def __init__(self, actor_critic):
        self.actor_critic = actor_critic  # actor_criticはクラスNetのディープ・ニューラルネットワーク
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.01)

    def update(self, rollouts):
        '''Advantageで計算した5つのstepの全てを使って更新します'''
        obs_shape = rollouts.observations.size()[2:]  # torch.Size([4, 84, 84])
        num_steps = NUM_ADVANCED_STEP
        num_processes = NUM_PROCESSES

        values, action_log_probs, entropy = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1, 4),
            rollouts.actions.view(-1, 1))

        # 注意：各変数のサイズ
        # rollouts.observations[:-1].view(-1, 4) torch.Size([80, 4])
        # rollouts.actions.view(-1, 1) torch.Size([80, 1])
        # values torch.Size([80, 1])
        # action_log_probs torch.Size([80, 1])
        # entropy torch.Size([])

        values = values.view(num_steps, num_processes,
                             1)  # torch.Size([5, 16, 1])
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        # advantage（行動価値-状態価値）の計算
        advantages = rollouts.returns[:-1] - values  # torch.Size([5, 16, 1])

        # Criticのlossを計算
        value_loss = advantages.pow(2).mean()

        # Actorのgainを計算、あとでマイナスをかけてlossにする
        action_gain = (action_log_probs*advantages.detach()).mean()
        # detachしてadvantagesを定数として扱う

        # 誤差関数の総和
        total_loss = (value_loss * value_loss_coef -
                      action_gain - entropy * entropy_coef)

        # 結合パラメータを更新
        self.actor_critic.train()  # 訓練モードに
        self.optimizer.zero_grad()  # 勾配をリセット
        total_loss.backward()  # バックプロパゲーションを計算
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm)
        #  一気に結合パラメータが変化しすぎないように、勾配の大きさは最大0.5までにする

        self.optimizer.step()  # 結合パラメータを更新


import copy


class Environment:
    def run(self):
        '''メインの実行'''

        # 同時実行する環境数分、envを生成
        envs = [gym.make(ENV) for i in range(NUM_PROCESSES)]

        # 全エージェントが共有して持つ頭脳Brainを生成
        n_in = envs[0].observation_space.shape[0]  # 状態は4
        n_out = envs[0].action_space.n  # 行動は2
        n_mid = 32
        actor_critic = Net(n_in, n_mid, n_out)  # ディープ・ニューラルネットワークの生成
        global_brain = Brain(actor_critic)

        # 格納用変数の生成
        obs_shape = n_in
        current_obs = torch.zeros(
            NUM_PROCESSES, obs_shape)  # torch.Size([16, 4])
        rollouts = RolloutStorage(
            NUM_ADVANCED_STEP, NUM_PROCESSES, obs_shape)  # rolloutsのオブジェクト
        episode_rewards = torch.zeros([NUM_PROCESSES, 1])  # 現在の試行の報酬を保持
        final_rewards = torch.zeros([NUM_PROCESSES, 1])  # 最後の試行の報酬を保持
        obs_np = np.zeros([NUM_PROCESSES, obs_shape])  # Numpy配列
        reward_np = np.zeros([NUM_PROCESSES, 1])  # Numpy配列
        done_np = np.zeros([NUM_PROCESSES, 1])  # Numpy配列
        each_step = np.zeros(NUM_PROCESSES)  # 各環境のstep数を記録
        episode = 0  # 環境0の試行数

        # 初期状態の開始
        obs = [envs[i].reset() for i in range(NUM_PROCESSES)]
        obs = np.array(obs)
        obs = torch.from_numpy(obs).float()  # torch.Size([16, 4])
        current_obs = obs  # 最新のobsを格納

        # advanced学習用のオブジェクトrolloutsの状態の1つ目に、現在の状態を保存
        rollouts.observations[0].copy_(current_obs)

        # 実行ループ
        for j in range(NUM_EPISODES*NUM_PROCESSES):  # 全体のforループ
            # advanced学習するstep数ごとに計算
            for step in range(NUM_ADVANCED_STEP):

                # 行動を求める
                with torch.no_grad():
                    action = actor_critic.act(rollouts.observations[step])

                # (16,1)→(16,)→tensorをNumPyに
                actions = action.squeeze(1).numpy()

                # 1stepの実行
                for i in range(NUM_PROCESSES):
                    obs_np[i], reward_np[i], done_np[i], _ = envs[i].step(
                        actions[i])

                    # episodeの終了評価と、state_nextを設定
                    if done_np[i]:  # ステップ数が200経過するか、一定角度以上傾くとdoneはtrueになる

                        # 環境0のときのみ出力
                        if i == 0:
                            print('%d Episode: Finished after %d steps' % (
                                episode, each_step[i]+1))
                            episode += 1

                        # 報酬の設定
                        if each_step[i] < 195:
                            reward_np[i] = -1.0  # 途中でこけたら罰則として報酬-1を与える
                        else:
                            reward_np[i] = 1.0  # 立ったまま終了時は報酬1を与える

                        each_step[i] = 0  # step数のリセット
                        obs_np[i] = envs[i].reset()  # 実行環境のリセット

                    else:
                        reward_np[i] = 0.0  # 普段は報酬0
                        each_step[i] += 1

                # 報酬をtensorに変換し、試行の総報酬に足す
                reward = torch.from_numpy(reward_np).float()
                episode_rewards += reward

                # 各実行環境それぞれについて、doneならmaskは0に、継続中ならmaskは1にする
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done_np])

                # 最後の試行の総報酬を更新する
                final_rewards *= masks  # 継続中の場合は1をかけ算してそのまま、done時には0を掛けてリセット
                # 継続中は0を足す、done時にはepisode_rewardsを足す
                final_rewards += (1 - masks) * episode_rewards

                # 試行の総報酬を更新する
                episode_rewards *= masks  # 継続中のmaskは1なのでそのまま、doneの場合は0に

                # 現在の状態をdone時には全部0にする
                current_obs *= masks

                # current_obsを更新
                obs = torch.from_numpy(obs_np).float()  # torch.Size([16, 4])
                current_obs = obs  # 最新のobsを格納

                # メモリオブジェクトに今stepのtransitionを挿入
                rollouts.insert(current_obs, action.data, reward, masks)

            # advancedのfor loop終了

            # advancedした最終stepの状態から予想する状態価値を計算

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.observations[-1]).detach()
                # rollouts.observationsのサイズはtorch.Size([6, 16, 4])

            # 全stepの割引報酬和を計算して、rolloutsの変数returnsを更新
            rollouts.compute_returns(next_value)

            # ネットワークとrolloutの更新
            global_brain.update(rollouts)
            rollouts.after_update()

            # 全部のNUM_PROCESSESが200step経ち続けたら成功
            if final_rewards.sum().numpy() >= NUM_PROCESSES:
                print('連続成功')
                break

# main学習
cartpole_env = Environment()
cartpole_env.run()














