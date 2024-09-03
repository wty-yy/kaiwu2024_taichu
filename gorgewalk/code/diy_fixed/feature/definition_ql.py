#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :gorge_walk
@File    :definition.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""


from kaiwu_agent.utils.common_func import create_cls, attached


SampleData = create_cls("SampleData", state=None, action=None, reward=None, next_state=None)


ObsData = create_cls("ObsData", feature=None)


ActData = create_cls("ActData", act=None)


@attached
def observation_process(raw_obs):
    pos = int(raw_obs[0])
    treasure_status = [int(item) for item in raw_obs[-10:]]
    state = 1024 * pos + sum([treasure_status[i] * (2**i) for i in range(10)])

    return ObsData(feature=int(state))


@attached
def action_process(act_data):
    return act_data.act


@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


def reward_shaping(frame_no, score, terminated, truncated, obs, _obs):
    reward = 0

    # The reward for winning
    # 奖励1. 获胜的奖励
    if terminated:
        reward += score
    """
    # The reward for being close to the finish line
    # 奖励2. 靠近终点的奖励:
    
    """
    # The reward for obtaining a treasure chest
    # 奖励3. 获得宝箱的奖励
    if score > 0 and not terminated:
        reward += score
    """
    # The reward for being close to the treasure chest (considering only the nearest one)
    # 奖励4. 靠近宝箱的奖励(只考虑最近的那个宝箱)
    
    """

    return reward
