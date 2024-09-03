from kaiwu_agent.agent.protocol.protocol import observation_process, action_process, sample_process
from diy.feature.definition import SecretRealmEnv
from kaiwu_agent.utils.common_func import attached
from diy.config import args
import torch
import random
import numpy as np
from diy.algorithm.agent import Agent

@attached
def workflow(envs, agents: list[Agent], logger=None, monitor=None):
    env, agent = envs[0], agents[0]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    env = SecretRealmEnv(env)
    agent.get_env(env)
    agent.train()
    agent.save_model()
