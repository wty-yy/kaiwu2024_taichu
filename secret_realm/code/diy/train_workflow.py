from kaiwu_agent.agent.protocol.protocol import observation_process, action_process, sample_process
from diy.feature.definition import SecretRealmEnv, ObsData
from kaiwu_agent.utils.common_func import attached
import time, os
from diy.config import args
import torch
import random
import numpy as np
from diy.algorithm.agent import Agent, init_writer
from kaiwu_agent.utils.common_func import Frame
from diy.utils.ckpt_manager import clean_ckpt_memory

@attached
def workflow(envs, agents: list[Agent], logger=None, monitor=None):
  env, agent = envs[0], agents[0]

  env = SecretRealmEnv(env, logger)
  args.num_iterations = (args.total_timesteps - 1) // (args.num_envs * args.num_steps) + 1
  logger.info(f"{args.num_iterations=}, {args.num_steps=}, {args.num_envs=}")
  start_time = time.time()
  last_clean_ckpt_time = time.time()
  writer = init_writer('aisrv')

  global_step = 0
  next_obs = env.reset()
  next_done = 0

  obs = np.zeros((args.num_steps, args.obs_dim), np.float32)
  actions = np.zeros(args.num_steps, np.float32)
  rewards = np.zeros(args.num_steps, np.float32)
  logprobs = np.zeros(args.num_steps, np.float32)
  dones = np.zeros(args.num_steps, np.float32)
  for iteration in range(1, args.num_iterations + 1):
    agent.load_model(id='latest')
    for step in range(args.num_steps):
      global_step += 1
      obs[step] = next_obs
      dones[step] = next_done

      act_data = agent.predict([ObsData(feature=next_obs)])[0]
      action, logprob = act_data.act, act_data.logprob
      actions[step] = action
      logprobs[step] = logprob

      next_obs, reward, terminations, truncations, infos = env.step(action)

      next_done = int(terminations | truncations)
      rewards[step] = reward
      if env.done:
        logger.info(f"pid={os.getpid()} End episode: gloabl_step={global_step}, episodic_reward={env.total_reward:.2f}, " +
                 f"episodic_score={int(env.total_score)}, " +
                 f"miss_treasure={env.miss_treasure}, n_treasure={env.n_treasure}")
        writer.add_scalar("charts/episodic_reward", env.total_reward, global_step)
        writer.add_scalar("charts/episodic_score", env.total_score, global_step)
        writer.add_scalar("charts/episodic_length", env.n_step, global_step)
        writer.add_scalar("charts/hit_wall", env.total_hit_wall, global_step)
        writer.add_scalar("charts/miss_treasure", env.miss_treasure, global_step)
        writer.add_scalar("charts/n_treasure", env.n_treasure, global_step)
        writer.add_scalar("charts/total_flash", env.total_flash, global_step)
        writer.add_scalar("charts/miss_buffer", env.miss_buffer, global_step)
        # self.monitor.put_data({os.getpid(): {
        #   'episodic_reward': self.env.total_reward,
        #   'episodic_score': self.env.total_score,
        #   'episodic_length': self.env.n_step,
        #   'hit_wall': self.env.total_hit_wall,
        #   'miss_treasure': self.env.miss_treasure,
        #   'n_treasure': self.env.n_treasure,
        # }})

    agent.learn(sample_process([Frame(
      obs=obs,
      actions=actions,
      rewards=rewards,
      dones=dones,
      logprobs=logprobs,
      next_obs=next_obs,
      next_done=next_done,
    )]))  # upload trajectory to buffer
    SPS = int(global_step / (time.time() - start_time))
    time_left = (args.num_iterations * args.num_steps - global_step) / SPS
    logger.info(f"global_step={global_step}, left_step={args.num_iterations*args.num_steps-global_step}")
    writer.add_scalar("time/SPS", SPS, global_step)
    writer.add_scalar("time/time_left", time_left, global_step)

    now = time.time()
    if now - last_clean_ckpt_time > 120:
      clean_ckpt_memory(path_name='user', logger=logger)
      last_clean_ckpt_time = now
      
  # agent.train()
  agent.save_model()
