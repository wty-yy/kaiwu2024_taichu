#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :back_to_the_realm
@File    :agent.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""

import torch
from kaiwudrl.common.config.config_control import CONFIG

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
import time
import math
from dqn.model.model import Model
from dqn.feature.definition import ActData
import numpy as np
from kaiwu_agent.agent.base_agent import (
    BaseAgent,
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
)
from kaiwu_agent.utils.common_func import attached
from dqn.config import Config

def colorstr(*input):
  # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
  *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
  colors = {
    'black': '\033[30m',  # basic colors
    'red': '\033[31m',
    'green': '\033[32m',
    'yellow': '\033[33m',
    'blue': '\033[34m',
    'magenta': '\033[35m',
    'cyan': '\033[36m',
    'white': '\033[37m',
    'bright_black': '\033[90m',  # bright colors
    'bright_red': '\033[91m',
    'bright_green': '\033[92m',
    'bright_yellow': '\033[93m',
    'bright_blue': '\033[94m',
    'bright_magenta': '\033[95m',
    'bright_cyan': '\033[96m',
    'bright_white': '\033[97m',
    'end': '\033[0m',  # misc
    'bold': '\033[1m',
    'underline': '\033[4m'}
  return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

import traceback
debug_colors = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan']
debug_idxs = 0
def show_debug(info, ptext="[DEBUG]", render_cfg=None, verbose_depth=1, show=True):
  def from_pos(depth=1):
    tmp = frame_summary[-1-depth]
    return f"<{tmp.filename}, line {tmp.lineno}> in {tmp.name} "

  frame_summary = traceback.extract_stack()
  verbose_depth = min(verbose_depth, len(frame_summary)-1)
  if not isinstance(info, tuple):
     info = (info,)
  if render_cfg is None:
    global debug_idxs
    render_cfg = (debug_colors[debug_idxs], 'bold')
    debug_idxs = (debug_idxs + 1) % len(debug_colors)

  s = ""
  for i in range(verbose_depth):
    tmp = ' FROM ' if i == 0 else ' IN   '
    s += colorstr(*render_cfg, f"{ptext+tmp+from_pos(i+1):+<120}") + '\n'
  for i in info:
    s += str(i) + ' '
  s += '\n'
  s += colorstr(*render_cfg, f"{ptext+' END  '+from_pos(1):-<120}")
  if show: print(s)
  return s


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.act_shape = Config.DIM_OF_ACTION_DIRECTION + Config.DIM_OF_TALENT
        self.direction_space = Config.DIM_OF_ACTION_DIRECTION
        self.talent_direction = Config.DIM_OF_TALENT
        self.obs_shape = Config.DIM_OF_OBSERVATION
        self.epsilon = Config.EPSILON
        self.egp = Config.EPSILON_GREEDY_PROBABILITY
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
        self.train_step = 0
        self.predict_count = 0
        self.last_report_monitor_time = 0

        self.agent_type = agent_type
        self.logger = logger
        self.monitor = monitor
        self.last_save = 0

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
        # self.logger.info(show_debug(f"pid={os.getpid()}, Predict", verbose_depth=100, show=False))
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

        model = getattr(self, "model")
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
        if time.time() - self.last_save > 10:
            self.last_save = time.time()
            # self.logger.info(f"{CONFIG.restore_dir=}, {CONFIG.user_ckpt_dir=}")
            # 保存到模型池中, 可以用来同步
            # self.save_model(path=f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/", source='framework')
            # 清理无用的模型, 避免内存爆炸
            from pathlib import Path
            path_tmp_dir = Path(f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/")
            files = sorted(list(path_tmp_dir.glob('model.ckpt-*')), key=lambda x: int(str(x).rsplit('.', 1)[0].rsplit('-', 1)[1]))
            self.logger.info(f"tmp weights number={len(files)}")
            for i, p in enumerate(files[:-1]):
                self.logger.info(f"remove weights rank={i+1}, {p}")
                p.unlink()

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
        # self.logger.info(show_debug("save_model", verbose_depth=100, show=False))

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        self.logger.info(f"pid={os.getpid()}, load model {model_file_path} successfully")
        # self.logger.info(show_debug(f"load model {model_file_path} successfully", verbose_depth=100, show=False))