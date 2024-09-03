#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :back_to_the_realm
@File    :back_to_the_realm_action.py
@Author  :kaiwu
@Date    :2022/10/20 11:43

"""

import numpy as np
from kaiwudrl.interface.array_spec import ArraySpec
from kaiwudrl.common.algorithms.distribution import CategoricalDist
from kaiwudrl.interface.action import Action, ActionSpec
from kaiwudrl.common.config.config_control import CONFIG

try:
    # 动态构建模块路径并导入 Config 类
    config_module = __import__(f"{CONFIG.algo}.config", fromlist=["Config"])
    Config = getattr(config_module, "Config")
except ModuleNotFoundError:
    raise NotImplementedError(f"The algorithm '{CONFIG.algo}' is not yet implemented")
except AttributeError:
    raise ImportError(f"The module '{CONFIG.algo}.config' does not have a 'Config' class")


class BackToTheRealmAction(Action):
    def __init__(self, a):
        self.a = a

    def get_action(self):
        return {"a": self.a}

    @staticmethod
    def action_space():
        direction_space = Config.DIM_OF_ACTION_DIRECTION
        talent_space = Config.DIM_OF_TALENT
        return {
            "a": ActionSpec(
                ArraySpec((direction_space + talent_space), np.int32),
                pdclass=CategoricalDist,
            )
        }

    def __str__(self):
        return str(self.a)
