#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :gorge_walk
@File    :train_workflow.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""

from kaiwu_agent.utils.common_func import Frame
from kaiwu_agent.utils.common_func import attached
import time
import math
from diy.feature.definition import (
    observation_process,
    action_process,
    sample_process,
    reward_shaping,
)
import os


@attached
def workflow(envs, agents, logger=None, monitor=None):
    env, agent = envs[0], agents[0]
    # EPISODES = 10000
    EPISODES = 1

    # Initializing monitoring data
    # 监控数据初始化
    monitor_data = {
        "reward": 0,
        "diy_1": 0,
        "diy_2": 0,
        "diy_3": 0,
        "diy_4": 0,
        "diy_5": 0,
    }
    last_report_monitor_time = time.time()

    logger.info("Start Training ...")
    start_t = time.time()

    total_rew, win_cnt = (
        0,
        0,
    )
    for episode in range(EPISODES):
        # User-defined environment launch configuration
        # 用户自定义的环境启动配置
        usr_conf = {
            "diy": {
                "start": [29, 9],
                "end": [11, 55],
                "treasure_id": [0, 1, 2, 3, 4],
                # "treasure_num": 5,
                # "treasure_random": 0,
                # "max_step": 2000,
            }
        }

        # Reset the environment and obtain the initial state
        # 重置环境, 并获取初始状态
        obs = env.reset(usr_conf=usr_conf)

        # Disaster recovery
        # 容灾
        if obs is None:
            continue

        # First frame processing
        # 首帧处理
        obs_data = observation_process(obs)

        # Task loop
        # 任务循环
        done = False
        while not done:
            act_data = agent.predict(list_obs_data=[obs_data])[0]
            act = action_process(act_data)
            frame_no, _obs, score, terminated, truncated, env_info = env.step(act)
            if _obs is None:
                break
            _obs_data = observation_process(_obs)
            reward = reward_shaping(frame_no, score, terminated, truncated, obs, _obs)
            done = terminated or truncated
            if terminated:
                win_cnt += 1
            frame = Frame(
                state=obs_data.feature,
                action=act,
                reward=reward,
                next_state=_obs_data.feature,
            )
            sample = sample_process([frame])
            agent.learn(sample)
            total_rew += reward
            obs_data = _obs_data

        # Reporting training progress
        # 上报训练进度
        # now = time.time()
        # if now - last_report_monitor_time > 60:
        #     logger.info(f"Episode: {episode + 1}, Reward: {total_rew}")
        #     logger.info(f"Training Win Rate: {win_cnt / (episode + 1)}")
        #     monitor_data["reward"] = total_rew
        #     if monitor:
        #         monitor.put_data({os.getpid(): monitor_data})

        #     total_rew = 0
        #     last_report_monitor_time = now

        # # The model has converged, training is complete, and reporting monitoring metric
        # # 模型收敛, 结束训练, 上报监控指标
        # if win_cnt / (episode + 1) > 0.9 and episode > 100:
        #     logger.info(f"Training Converged at Episode: {episode + 1}")
        #     monitor_data["reward"] = total_rew
        #     if monitor:
        #         monitor.put_data({os.getpid(): monitor_data})
        #     break

    # end_t = time.time()
    # logger.info(f"Training Time for {episode + 1} episodes: {end_t - start_t} s")
    # agent.episodes = episode + 1

    # model saving
    # 保存模型
    agent.save_model()

    return
