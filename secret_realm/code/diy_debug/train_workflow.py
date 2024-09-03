import time
import numpy as np
from kaiwu_agent.utils.common_func import Frame
from kaiwu_agent.utils.common_func import attached
from diy.feature.definition import (
    observation_process,
    action_process,
    sample_process,
    SecretRealmEnv
)

@attached
def workflow(envs, agents, logger=None, monitor=None):
    env, agent = envs[0], agents[0]
    logger.info("HI")

    env = SecretRealmEnv(env, logger)
    for episode in range(1):
        done = False
        obs = env.reset()
        grid_pos, acc_pos = obs['grid_pos'], obs['norm_pos'] * 64000
        delta_grid_pos, delta_acc_pos = [], []
        for i in range(40):
            if i < 29:
                action = [2, 0]
            if i in list(range(29, 50)):
                action = [0, 1]
            if i in list(range(50, 100)):
                action = [1, 0]
            if i in list(range(100, 120)):
                action = [5, 0]
            # if i in list(range(40, 50)):
            #     action = [7, 0]
            obs, reward, done, info = env.step(action)
            _grid_pos, _acc_pos = obs['grid_pos'], obs['norm_pos'] * 64000
            delta_grid_pos.append(np.linalg.norm(grid_pos - _grid_pos))
            delta_acc_pos.append(np.linalg.norm(acc_pos - _acc_pos))
            grid_pos, acc_pos = _grid_pos, _acc_pos
        print(f"{delta_grid_pos=}\n{np.mean(delta_grid_pos)=}")
        print(f"{delta_acc_pos=}\n{np.mean(delta_acc_pos)=}")
        # agent.load_model(id="latest")

    # agent.save_model()
