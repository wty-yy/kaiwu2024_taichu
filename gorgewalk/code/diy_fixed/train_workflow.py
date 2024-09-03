from kaiwu_agent.utils.common_func import Frame
from kaiwu_agent.utils.common_func import attached
from diy.feature.definition import (
  observation_process,
  action_process,
  sample_process,
)
from diy.config import Config
from diy.utils import show_debug
import os, time


@attached
def workflow(envs, agents, logger=None, monitor=None):
  for i in range(20000):
    if i % 1000 == 0:
      print(f"{i=}")
    agents[0].learn([None])
  exit()
  env, agent = envs[0], agents[0]
  agent.verbose = False

  episodes = 5
  for _ in range(episodes):
    history = {
      'reward': 0,
      'n_steps': 0,
    }
    agent.reset()
    s = env.reset(Config.ENV_CFG)
    while s is None:
      s = env.reset(Config.ENV_CFG)
    s = observation_process(s)
    done = False
    while not done:
      # a = action_process(agent.predict([s])[0])  # predict -> ActData
      a = action_process(agent.exploit([s])[0])  # predict -> ActData
      _, s, r, terminal, truncated, _ = env.step(a)
      s = observation_process(s)  # ObsData
      history['reward'] += r; history['n_steps'] += 1
      done = terminal | truncated

      # rubbish
      sample = sample_process([Frame(a=None)])  # SampleData
      agent.learn(sample)
    logger.info(f"{history['reward']=}, {history['n_steps']=}")
    # if monitor:
    #   monitor.put_data({os.getpid(): history})
  agent.save_model()
