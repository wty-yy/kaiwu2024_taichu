#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :back_to_the_realm
@File    :train_workflow.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""

import time
from kaiwu_agent.utils.common_func import Frame
from kaiwu_agent.utils.common_func import attached
from target_dqn.feature.definition import (
    observation_process,
    action_process,
    sample_process,
    reward_shaping,
)
from conf.usr_conf import usr_conf_check


@attached
def workflow(envs, agents, logger=None, monitor=None):
    env, agent = envs[0], agents[0]
    epoch_num = 100000
    episode_num_every_epoch = 1
    g_data_truncat = 256
    last_save_model_time = 0

    # User-defined game start configuration
    # 用户自定义的游戏启动配置
    usr_conf = {
        "diy": {
            "start": 2,
            "end": 1,
            # "treasure_id": [4, 5, 6, 7, 8, 9],
            "treasure_random": 1,
            "talent_type": 1,
            "treasure_num": 8,
            "max_step": 2000,
        }
    }

    # usr_conf_check is a tool to check whether the game configuration is correct
    # It is recommended to perform a check before calling reset.env
    # usr_conf_check会检查游戏配置是否正确，建议调用reset.env前先检查一下
    valid = usr_conf_check(usr_conf, logger)
    if not valid:
        logger.error(f"usr_conf_check return False, please check")
        return

    for epoch in range(epoch_num):
        epoch_total_rew = 0
        data_length = 0
        for g_data in run_episodes(episode_num_every_epoch, env, agent, g_data_truncat, usr_conf, logger):
            data_length += len(g_data)
            total_rew = sum([i.rew for i in g_data])
            epoch_total_rew += total_rew
            agent.learn(g_data)
            g_data.clear()

        avg_step_reward = 0
        if data_length:
            avg_step_reward = f"{(epoch_total_rew/data_length):.2f}"

        # save model file
        # 保存model文件
        now = time.time()
        if now - last_save_model_time >= 120:
            agent.save_model()
            last_save_model_time = now

        logger.info(f"Avg Step Reward: {avg_step_reward}, Epoch: {epoch}, Data Length: {data_length}")


def run_episodes(n_episode, env, agent, g_data_truncat, usr_conf, logger):
    for episode in range(n_episode):
        collector = list()

        # Reset the game and get the initial state
        # 重置游戏, 并获取初始状态
        obs = env.reset(usr_conf=usr_conf)
        env_info = None

        # Disaster recovery
        # 容灾
        if obs is None:
            continue

        # At the start of each game, support loading the latest model file
        # The call will load the latest model from a remote training node
        # 每次对局开始时, 支持加载最新model文件, 该调用会从远程的训练节点加载最新模型
        agent.load_model(id="latest")

        # Feature processing
        # 特征处理
        obs_data = observation_process(obs)

        done = False
        step = 0
        bump_cnt = 0

        while not done:
            # Agent performs inference, gets the predicted action for the next frame
            # Agent 进行推理, 获取下一帧的预测动作
            act_data = agent.predict(list_obs_data=[obs_data])[0]

            # Unpack ActData into action
            # ActData 解包成动作
            act = action_process(act_data)

            # Interact with the environment, execute actions, get the next state
            # 与环境交互, 执行动作, 获取下一步的状态
            frame_no, _obs, score, terminated, truncated, _env_info = env.step(act)
            if _obs is None:
                break

            step += 1

            # Feature processing
            # 特征处理
            _obs_data = observation_process(_obs, _env_info)

            # Disaster recovery
            # 容灾
            if truncated and frame_no is None:
                break

            treasures_num = 0
            # Calculate reward
            # 计算 reward
            if env_info is None:
                reward = 0
            else:
                reward, is_bump = reward_shaping(
                    frame_no,
                    score,
                    terminated,
                    truncated,
                    obs,
                    _obs,
                    env_info,
                    _env_info,
                )
                treasure_dists = [pos.grid_distance for pos in _obs.feature.treasure_pos]
                treasures_num = treasure_dists.count(1.0)
                # Wall bump behavior statistics
                # 撞墙行为统计
                bump_cnt += is_bump

            # Determine game over, and update the number of victories
            # 判断游戏结束, 并更新胜利次数
            if truncated:
                logger.info(
                    f"truncated is True, so this episode {episode} timeout, \
                        collected treasures: {treasures_num  - 7}"
                )
            elif terminated:
                logger.info(
                    f"terminated is True, so this episode {episode} reach the end, \
                        collected treasures: {treasures_num  - 7}"
                )
            done = terminated or truncated

            # Construct game frames to prepare for sample construction
            # 构造游戏帧，为构造样本做准备
            frame = Frame(
                obs=obs_data.feature,
                _obs=_obs_data.feature,
                obs_legal=obs_data.legal_act,
                _obs_legal=_obs_data.legal_act,
                act=act,
                rew=reward,
                done=done,
                ret=reward,
            )

            collector.append(frame)

            # If the number of game frames reaches the threshold, the sample is processed and sent to training
            # 如果游戏帧数达到阈值，则进行样本处理，将样本送去训练
            if len(collector) % g_data_truncat == 0:
                collector = sample_process(collector)
                yield collector

            # If the game is over, the sample is processed and sent to training
            # 如果游戏结束，则进行样本处理，将样本送去训练
            if done:
                if len(collector) > 0:
                    collector = sample_process(collector)
                    yield collector
                break

            # Status update
            # 状态更新
            obs_data = _obs_data
            obs = _obs
            env_info = _env_info
