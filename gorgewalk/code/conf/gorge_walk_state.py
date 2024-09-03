#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :gorge_walk
@File    :gorge_walk_state.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""

import numpy as np
from kaiwudrl.interface.array_spec import ArraySpec
from kaiwudrl.interface.state import State
from kaiwudrl.common.config.config_control import CONFIG

try:
    # 动态构建模块路径并导入 Config 类
    config_module = __import__(f"{CONFIG.algo}.config", fromlist=["Config"])
    Config = getattr(config_module, "Config")
except ModuleNotFoundError:
    raise NotImplementedError(f"The algorithm '{CONFIG.algo}' is not yet implemented")
except AttributeError:
    raise ImportError(f"The module '{CONFIG.algo}.config' does not have a 'Config' class")

"""
主要用于actor上使用
"""


class GorgeWalkState(State):
    def __init__(self, value):
        """
        Args:
            value: 由run_handler构造本类, 为on_update函数的一个返回值(当需要预测时)
        """
        self.value = value

    def get_state(self):
        """
        根据构造函数中传入的value,构造返回一个dict
        dict会传给Actor进行预测
        """
        observation = np.array(self.value["observation"], dtype=np.float64)
        legal_action = np.array(self.value["legal_action"], dtype=np.float64)
        sub_action_mask = np.array(self.value["sub_action_mask"], dtype=np.float64)
        lstm_hidden = np.array(self.value["lstm_hidden"], dtype=np.float64)
        lstm_cell = np.array(self.value["lstm_cell"], dtype=np.float64)
        return {
            "observation": observation,
            "legal_action": legal_action,
            "sub_action_mask": sub_action_mask,
            "lstm_hidden": lstm_hidden,
            "lstm_cell": lstm_cell,
        }

    @staticmethod
    def state_space():
        """
        定义观测空间维度和数据结构, 规定state中每个变量的shape, 必须为numpy数组
        """
        observation_shape = (Config.OBSERVATION_SHAPE,)
        legal_action_shape = (Config.LEGAL_ACTION_SHAPE,)
        sub_action_mask_shape = (Config.SUB_ACTION_MASK_SHAPE,)
        lstm_hidden_shape = (Config.LSTM_HIDDEN_SHAPE,)
        lstm_cell_shape = (Config.LSTM_CELL_SHAPE,)
        return {
            "observation": ArraySpec(observation_shape, np.float64),
            "legal_action": ArraySpec(legal_action_shape, np.float64),
            "sub_action_mask": ArraySpec(sub_action_mask_shape, np.float64),
            "lstm_hidden": ArraySpec(lstm_hidden_shape, np.float64),
            "lstm_cell": ArraySpec(lstm_cell_shape, np.float64),
        }

    def __str__(self):
        return str(self.value)
