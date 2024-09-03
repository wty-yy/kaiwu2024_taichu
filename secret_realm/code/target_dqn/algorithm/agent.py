#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :back_to_the_realm
@File    :agent.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
import time
from target_dqn.model.model import Model
from target_dqn.feature.definition import ActData
import numpy as np
from copy import deepcopy
from kaiwu_agent.agent.base_agent import (
    BaseAgent,
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
)
from kaiwu_agent.utils.common_func import attached
from target_dqn.config import Config


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.act_shape = Config.DIM_OF_ACTION_DIRECTION + Config.DIM_OF_TALENT
        self.direction_space = Config.DIM_OF_ACTION_DIRECTION
        self.talent_direction = Config.DIM_OF_TALENT
        self.obs_shape = Config.DIM_OF_OBSERVATION
        self.epsilon = Config.EPSILON
        self.egp = Config.EPSILON_GREEDY_PROBABILITY
        self.target_update_freq = Config.TARGET_UPDATE_FREQ
        self.obs_split = Config.DESC_OBS_SPLIT
        self._gamma = Config.GAMMA
        self.lr = Config.START_LR

        self.device = device
        self.model = Model(
            state_shape=self.obs_shape,
            action_shape=self.act_shape,
            softmax=False,
        )
        self.model.to(self.device)
        self.optim = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.target_model = deepcopy(self.model)
        self.train_step = 0
        self.predict_count = 0
        self.last_report_monitor_time = 0

        self.agent_type = agent_type
        self.logger = logger
        self.monitor = monitor

    def __convert_to_tensor(self, data):
        if isinstance(data, list):
            return torch.tensor(
                np.array(data),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            return torch.tensor(
                data,
                device=self.device,
                dtype=torch.float32,
            )

    def __predict_detail(self, list_obs_data, exploit_flag=False):
        batch = len(list_obs_data)
        feature_vec = [obs_data.feature[: self.obs_split[0]] for obs_data in list_obs_data]
        feature_map = [obs_data.feature[self.obs_split[0] :] for obs_data in list_obs_data]
        legal_act = [obs_data.legal_act for obs_data in list_obs_data]
        legal_act = torch.tensor(np.array(legal_act))
        legal_act = (
            torch.cat(
                (
                    legal_act[:, 0].unsqueeze(1).expand(batch, self.direction_space),
                    legal_act[:, 1].unsqueeze(1).expand(batch, self.talent_direction),
                ),
                1,
            )
            .bool()
            .to(self.device)
        )
        model = self.model
        model.eval()
        # Exploration factor, starting with an initial epsilon of 0.5,
        # we want epsilon to decrease as the number of prediction steps increases, until it reaches 0.1
        # 探索因子, 初始epsilon为0.5，我们希望epsilon随着预测步数越来越小，直到0.1为止
        self.epsilon = max(0.1, 0.5 - self.predict_count / self.egp)

        with torch.no_grad():
            # epsilon greedy
            if not exploit_flag and np.random.rand(1) < self.epsilon:
                random_action = np.random.rand(batch, self.act_shape)
                random_action = torch.tensor(random_action, dtype=torch.float32).to(self.device)
                random_action = random_action.masked_fill(~legal_act, 0)
                act = random_action.argmax(dim=1).cpu().view(-1, 1).tolist()
            else:
                feature = [
                    self.__convert_to_tensor(feature_vec),
                    self.__convert_to_tensor(feature_map).view(batch, *self.obs_split[1]),
                ]
                logits, _ = model(feature, state=None)
                logits = logits.masked_fill(~legal_act, float(torch.min(logits)))
                act = logits.argmax(dim=1).cpu().view(-1, 1).tolist()

        format_action = [[instance[0] % self.direction_space, instance[0] // self.direction_space] for instance in act]
        self.predict_count += 1
        return [ActData(move_dir=i[0], use_talent=i[1]) for i in format_action]

    @predict_wrapper
    def predict(self, list_obs_data):
        return self.__predict_detail(list_obs_data, exploit_flag=False)

    @exploit_wrapper
    def exploit(self, list_obs_data):
        return self.__predict_detail(list_obs_data, exploit_flag=True)

    @learn_wrapper
    def learn(self, list_sample_data):

        t_data = list_sample_data
        batch = len(t_data)

        # [b, d]
        batch_feature_vec = [frame.obs[: self.obs_split[0]] for frame in t_data]
        batch_feature_map = [frame.obs[self.obs_split[0] :] for frame in t_data]
        batch_action = torch.LongTensor(np.array([int(frame.act) for frame in t_data])).view(-1, 1).to(self.device)

        _batch_obs_legal = torch.tensor(np.array([frame._obs_legal for frame in t_data]))
        _batch_obs_legal = (
            torch.cat(
                (
                    _batch_obs_legal[:, 0].unsqueeze(1).expand(batch, self.direction_space),
                    _batch_obs_legal[:, 1].unsqueeze(1).expand(batch, self.talent_direction),
                ),
                1,
            )
            .bool()
            .to(self.device)
        )

        rew = torch.tensor(np.array([frame.rew for frame in t_data]), device=self.device)
        _batch_feature_vec = [frame._obs[: self.obs_split[0]] for frame in t_data]
        _batch_feature_map = [frame._obs[self.obs_split[0] :] for frame in t_data]
        not_done = torch.tensor(np.array([0 if frame.done == 1 else 1 for frame in t_data]), device=self.device)

        batch_feature = [
            self.__convert_to_tensor(batch_feature_vec),
            self.__convert_to_tensor(batch_feature_map).view(batch, *self.obs_split[1]),
        ]
        _batch_feature = [
            self.__convert_to_tensor(_batch_feature_vec),
            self.__convert_to_tensor(_batch_feature_map).view(batch, *self.obs_split[1]),
        ]

        model = getattr(self, "target_model")
        model.eval()
        with torch.no_grad():
            q, h = model(_batch_feature, state=None)
            q = q.masked_fill(~_batch_obs_legal, float(torch.min(q)))
            q_max = q.max(dim=1).values.detach()

        target_q = rew + self._gamma * q_max * not_done

        self.optim.zero_grad()

        model = getattr(self, "model")
        model.train()
        logits, h = model(batch_feature, state=None)

        loss = torch.square(target_q - logits.gather(1, batch_action).view(-1)).mean()
        loss.backward()
        self.optim.step()

        self.train_step += 1

        # Update the target network
        # 更新target网络
        if self.train_step % self.target_update_freq == 0:
            self.update_target_q()

        value_loss = loss.detach().item()
        q_value = target_q.mean().detach().item()
        reward = rew.mean().detach().item()

        # Periodically report monitoring
        # 按照间隔上报监控
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            monitor_data = {
                "value_loss": value_loss,
                "q_value": q_value,
                "reward": reward,
                "diy_1": 0,
                "diy_2": 0,
                "diy_3": 0,
                "diy_4": 0,
                "diy_5": 0,
            }
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})

            self.last_report_monitor_time = now

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        # Copy the model's state dictionary to the CPU
        # 将模型的状态字典拷贝到CPU
        model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(model_state_dict_cpu, model_file_path)

        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(
            torch.load(model_file_path, map_location=self.device),
        )

        self.logger.info(f"load model {model_file_path} successfully")

    def update_target_q(self):
        self.target_model.load_state_dict(self.model.state_dict())
