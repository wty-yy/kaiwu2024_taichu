from kaiwu_agent.agent.protocol.protocol import observation_process, action_process, sample_process
from kaiwu_agent.utils.common_func import attached
from kaiwu_agent.utils.common_func import Frame
from diy.algorithm.agent import Agent
from typing import List
from diy.utils import ProcessPrinter
import numpy as np

n_episodes = 10

@attached
def workflow(envs, agents: List[Agent], logger=None, monitor=None):
    env, agent = envs[0], agents[0]
    process_printer = ProcessPrinter()

    for episode in range(n_episodes):
        env_cfg = {
            'diy': {
                'start': [29, 9],
                'end': [11, 55],
                'treasure_id': [0, 1, 2, 3, 4],
            }
        }
        obs = env.reset(env_cfg)
        if obs is None: continue
        obs_data = observation_process(obs)

        done = False
        timesteps = 0
        while not done:
            # 这里的动作与状态均只需考虑一个
            act_data = agent.predict(list_obs_data=[obs_data])[0]
            act = action_process(act_data)

            frame_no, _obs, score, terminal, truncated, env_info = env.step(act)
            if _obs is None:
                _obs = np.zeros_like(obs)
            _obs_data = observation_process(_obs)
            done = terminal or truncated
            reward = score

            """
            这里的格式保持和definition.py中的SampleData2NumpyData输入的g_data格式相同
            然后sample_process就会调用，SampleData2NumpyData转存到buffer中去
            我们无需调用agent.learn，因为在buffer到达一定比例后，
            learner进程会自动从buffer中采样，调用NumpyData2SampleData传入到agent.learn()中
            """
            frame = Frame(
                obs=obs_data,
                act=act,
                rew=reward,
                _obs=_obs_data,
                terminal=terminal,
            )
            sample = sample_process([frame])
            # agent.learn(sample)
            obs_data = _obs_data
            timesteps += 1
            logger.info(f"{episode=}, {timesteps=}, {done=}")
        
    return
