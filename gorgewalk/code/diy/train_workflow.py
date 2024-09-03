from kaiwu_agent.agent.protocol.protocol import observation_process, action_process, sample_process
from diy.feature.definition import GorgeWalk
from kaiwu_agent.utils.common_func import Frame
from kaiwu_agent.utils.common_func import attached
from diy.config import Args
import torch
import random
import argparse
import numpy as np
from diy.utils import show_debug

@attached
def workflow(envs, agents, logger=None, monitor=None):
    env, agent = envs[0], agents[0]

    args = argparse.Namespace(**Args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    env = GorgeWalk(env)
    agent.get_everything(args, env)
    agent.train()
    agent.save_model()
    return
