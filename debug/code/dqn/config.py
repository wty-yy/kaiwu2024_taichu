#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :back_to_the_realm
@File    :config.py
@Author  :kaiwu
@Date    :2023/7/1 10:37

"""


# Configuration
# 配置，包含维度设置，算法参数设置，文件的最后一些配置是开悟平台使用不要改动
class Config:

    # The input dimension of samples on the learner from Reverb varies depending on the algorithm used.
    # For instance, the dimension for DQN is 21624, while for target DQN, it is also 21624. Have to set
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样, 比如示例代码中dqn的维度是21624, target_dqn的维度是21624
    # **注意**，此项必须正确配置，应该与definition.py中的NumpyData2SampleData函数数据对齐，否则可能报样本维度错误
    SAMPLE_DIM = 21624

    # Size of observation
    # observation的维度，注意在我们的示例代码中原特征维度是10808，这里是经过CNN处理之后的维度与原始向量特征拼接后的维度
    DIM_OF_OBSERVATION = 128 + 404

    # Dimension of movement action direction
    # 移动动作方向的维度
    DIM_OF_ACTION_DIRECTION = 8

    # Dimension of flash action direction
    # 闪现动作方向的维度
    DIM_OF_TALENT = 8

    # 描述如何进行特征分割，示例代码中的特征处理成向量+特征图，以下配置描述了两者的维度
    # pos_float + pos_onehot + organ + cd&talent, obstacle_map, treasure_map, end_map, location_memory
    #         2 +   128*2    +  9*17 +     2,     51*51*4
    DESC_OBS_SPLIT = [404, (4, 51, 51)]  # sum = 10808

    # 以下是一些算法配置

    # Exploration factor, see the calculation of epsilon in the function in the above comment
    # 探索因子, epsilon的计算见上面注释中的函数
    EPSILON_GREEDY_PROBABILITY = 30000

    # Discount factor GAMMA in RL
    # RL中的回报折扣GAMMA
    GAMMA = 0.9

    # epsilon
    EPSILON = 0.9

    # Initial learning rate
    # 初始的学习率
    START_LR = 1e-4

    # Configuration about kaiwu usage. The following configurations can be ignored
    # 关于开悟平台使用的配置，是可以忽略的配置，不需要改动
    SUB_ACTION_MASK_SHAPE = 0
    LSTM_HIDDEN_SHAPE = 0
    LSTM_CELL_SHAPE = 0
    OBSERVATION_SHAPE = 4500
    LEGAL_ACTION_SHAPE = 2
