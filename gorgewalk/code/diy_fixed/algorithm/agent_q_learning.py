#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :gorge_walk
@File    :agent.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""

import numpy as np
from kaiwu_agent.agent.base_agent import (
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
)
from kaiwu_agent.utils.common_func import attached
from diy.feature.definition import ActData
from kaiwu_agent.agent.base_agent import BaseAgent
from diy.config import Config


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        self.logger = logger

        # Initialize parameters
        # 参数初始化
        self.state_size = Config.STATE_SIZE
        self.action_size = Config.ACTION_SIZE
        self.learning_rate = Config.LEARNING_RATE
        self.gamma = Config.GAMMA
        self.epsilon = Config.EPSILON
        self.episodes = Config.EPISODES

        # Reset the Q-table
        # 重置Q表
        self.Q = np.ones([self.state_size, self.action_size])

        super().__init__(agent_type, device, logger, monitor)

    @predict_wrapper
    def predict(self, list_obs_data):
        """
        The input is list_obs_data, and the output is list_act_data.
        """
        """
        输入是 list_obs_data, 输出是 list_act_data
        """
        state = list_obs_data[0].feature
        act = self._epsilon_greedy(state=state, epsilon=self.epsilon)

        return [ActData(act=act)]

    @exploit_wrapper
    def exploit(self, list_obs_data):
        state = list_obs_data[0].feature
        act = np.argmax(self.Q[state, :])

        return [ActData(act=act)]

    def _epsilon_greedy(self, state, epsilon=0.1):
        """
        Epsilon-greedy algorithm for action selection
        """
        """
        ε-贪心算法用于动作选择
        """
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, self.action_size)

        # Exploitation
        # 利用
        else:
            action = np.argmax(self.Q[state, :])

        return action

    @learn_wrapper
    def learn(self, list_sample_data):
        """
        Update the Q-table with the given game data:
            - list_sample: each sampple is [state, action, reward, new_state]
        Using the following formula to update q value:
            - Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        """
        """
        使用给定的数据更新Q表格:
        list_sample:每个样本是[state, action, reward, new_state]
        使用以下公式更新Q值:
        Q(s,a) := Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        其中：
        Q(s,a) 表示状态s下采取动作a的Q值
        lr 是学习率(learning rate), 用于控制每次更新的幅度
        R(s,a) 是在状态s下采取动作a所获得的奖励
        gamma 是折扣因子(discount factor), 用于平衡当前奖励和未来奖励的重要性
        max Q(s',a') 表示在新状态s'下采取所有可能动作a'的最大Q值
        """
        sample = list_sample_data[0]
        state, action, reward, next_state = (
            sample.state,
            sample.action,
            sample.reward,
            sample.next_state,
        )

        delta = reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[state, action]

        self.Q[state, action] += self.learning_rate * delta

        return

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"
        np.save(model_file_path, self.Q)
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"
        try:
            self.Q = np.load(model_file_path)
            self.logger.info(f"load model {model_file_path} successfully")
        except FileNotFoundError:
            self.logger.info(f"File {model_file_path} not found")
            exit(1)
