import time
from kaiwu_agent.utils.common_func import Frame
from kaiwu_agent.utils.common_func import attached
from diy.feature.definition import (
    observation_process,
    action_process,
    sample_process,
    reward_shaping,
)


@attached
def workflow(envs, agents, logger=None, monitor=None):
    env, agent = envs[0], agents[0]

    # At the start of each game, support loading the latest model file
    # 每次对局开始时, 支持加载最新model文件, 该调用会从远程的训练节点加载最新模型
    agent.load_model(id="latest")
    
    # model saving
    # 保存模型
    agent.save_model()

    return
