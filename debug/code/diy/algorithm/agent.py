import time, os
import numpy as np
from kaiwu_agent.agent.base_agent import BaseAgent
from kaiwu_agent.agent.base_agent import (
  learn_wrapper,
  save_model_wrapper,
  load_model_wrapper,
  predict_wrapper,
  exploit_wrapper,
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from diy.feature.definition import (
  ActData, SecretRealmEnv,
  sample_process
)
from diy.algorithm.model import Model
from diy.utils import show_time
from diy.config import args
from pathlib import Path
from kaiwu_agent.utils.common_func import Frame
from diy.utils.clean_ckpt_memory import clean_ckpt_memory
PATH_ROOT = Path(__file__).parents[2]
PATH_LOGS_DIR = PATH_ROOT / "log/tensorboard"
PATH_LOGS_DIR.mkdir(exist_ok=True, parents=True)

class Agent(BaseAgent):

  def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
    super().__init__(agent_type, device, logger, monitor)
    self.logger, self.monitor = logger, monitor
    logger.info(f"pid={os.getpid()}, {device=}")
    self.device = device
    self.model = Model().to(self.device)
    self.env: SecretRealmEnv = None

    run_name = f"secret_realm_ppo_v{args.version}_{agent_type}_pid{os.getpid()}"
    self.writer = SummaryWriter(str(PATH_LOGS_DIR / run_name))
    self.writer.add_text(
      "hyperparameters",
      "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    self.global_step = 0
    self.last_opt_time = None
    # if agent_type == 'learner':
    self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, eps=1e-5)
    self.start_time = None

  def get_env(self, env):
    self.env = env
  
  def _predict(self, list_obs_data):
    obs = list_obs_data[0].feature
    obs = torch.Tensor(obs).to(self.device).view(1, -1)
    action, _, _, _ = self.model.get_action_and_value(obs)
    return [ActData(act=int(action[0].cpu().numpy()))]

  @predict_wrapper
  def predict(self, list_obs_data):
    return self._predict(list_obs_data)

  @exploit_wrapper
  def exploit(self, list_obs_data):
    return self._predict(list_obs_data)

  @learn_wrapper
  def learn(self, batch_data):
    """
    Args:
      batch_data: [List[SampleData]]
        learner sample batch_data from buffer,
        len(batch_data)=train_batch_size (configure_app.toml).
    """
    if self.start_time is None:
      self.logger.info(f"batchsize={len(batch_data)}")
      self.start_time = self.last_opt_time = time.time()
    learn_start_time = time.time()
    args.batch_size = len(batch_data)
    args.minibatch_size = int(args.batch_size // args.num_minibatches) # 32
    # How to change learning_rate and ent_coef
    # flatten the batch
    list2tensor = lambda x: torch.tensor(np.array(x, np.float32)).to(self.device)
    b_obs = list2tensor([x.obs for x in batch_data])
    b_actions = list2tensor([x.action for x in batch_data])
    b_logprobs = list2tensor([x.logprob for x in batch_data])
    b_values = list2tensor([x.value for x in batch_data])
    b_advantages = list2tensor([x.advantage for x in batch_data])
    b_returns = list2tensor([x.ret for x in batch_data])
    # print(f"{b_obs.shape}, {b_actions.shape}, {b_logprobs.shape}, {b_values.shape}, {b_advantages.shape}, {b_returns.shape}")

    # Optimizing the policy and value network
    b_inds = np.arange(args.batch_size)
    clipfracs = []
    for epoch in range(args.update_epochs):
      np.random.shuffle(b_inds)
      for start in range(0, args.batch_size, args.minibatch_size):
        end = start + args.minibatch_size
        mb_inds = b_inds[start:end]
        _, newlogprob, entropy, newvalue = self.model.get_action_and_value(
          b_obs[mb_inds],
          b_actions.long()[mb_inds])
        logratio = newlogprob - b_logprobs[mb_inds]
        ratio = logratio.exp()

        with torch.no_grad():
          # calculate approx_kl http://joschu.net/blog/kl-approx.html
          old_approx_kl = (-logratio).mean()
          approx_kl = ((ratio - 1) - logratio).mean()
          clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

        mb_advantages = b_advantages[mb_inds]
        if args.norm_adv:
          mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() +
                                      1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef,
                            1 + args.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        if args.clip_vloss:
          v_loss_unclipped = (newvalue - b_returns[mb_inds])**2
          v_clipped = b_values[mb_inds] + torch.clamp(
            newvalue - b_values[mb_inds],
            -args.clip_coef,
            args.clip_coef,
          )
          v_loss_clipped = (v_clipped - b_returns[mb_inds])**2
          v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
          v_loss = 0.5 * v_loss_max.mean()
        else:
          v_loss = 0.5 * ((newvalue - b_returns[mb_inds])**2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
        self.optimizer.step()
        self.global_step += 1

      if args.target_kl is not None and approx_kl > args.target_kl:
        break

    # 计算方差解释率，衡量模型预测的准确性
    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    
    learn_SPS = self.global_step / (time.time() - self.start_time)
    # self.logger.info(f"SPS: {SPS}")
    self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
    self.writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
    self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
    self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
    self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.global_step)
    self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.global_step)
    self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.global_step)
    self.writer.add_scalar("losses/explained_variance", explained_var, self.global_step)
    self.writer.add_scalar("time/learn_SPS_avg", learn_SPS, self.global_step)
    self.writer.add_scalar("time/learn_SPS", (args.update_epochs * args.num_minibatches) / (time.time() - learn_start_time), self.global_step)
    # self.monitor.put_data({os.getpid(): {
    #   "learning_rate": self.optimizer.param_groups[0]["lr"],
    #   "value_loss": v_loss.item(),
    #   "policy_loss": pg_loss.item(),
    #   "entropy": entropy_loss.item(),
    #   "old_approx_kl": old_approx_kl.item(),
    #   "approx_kl": approx_kl.item(),
    #   "clipfrac": np.mean(clipfracs),
    #   "explained_variance": explained_var,
    #   "SPS": SPS,
    # }})

    now = time.time()
    if now - self.last_opt_time > 10:
      clean_ckpt_memory()
      self.last_opt_time = now

  def rollout(self):
    assert self.model is not None, "model is None"
    assert self.env is not None, "env is None"
    args.batch_size = int(args.num_envs * args.num_steps) # 128
    args.num_iterations = args.total_timesteps // args.batch_size
    self.start_time = time.time()
    self.last_opt_time = time.time()

    next_obs = self.env.reset()
    next_done = 0

    rewards = np.zeros(args.num_steps, np.float32)
    dones = np.zeros(args.num_steps, np.float32)
    values = np.zeros(args.num_steps, np.float32)
    for iteration in range(1, args.num_iterations + 1):
      collector = []
      self.load_model(id='latest')  # update model
      for step in range(0, args.num_steps):
        self.global_step += 1
        frame = Frame()
        collector.append(frame)
        frame.obs = next_obs
        dones[step] = next_done

        with torch.no_grad():
          next_obs = torch.Tensor(next_obs).to(self.device).view(1,-1)
          action, logprob, _, value = self.model.get_action_and_value(next_obs)
        values[step] = frame.value = value.cpu().numpy()[0,0]
        frame.action = action.cpu().numpy()[0]
        frame.logprob = logprob.cpu().numpy()[0]

        next_obs, reward, terminations, truncations, infos = self.env.step(action[0].cpu().numpy())

        next_done = np.logical_or(terminations, truncations)
        rewards[step] = reward
        next_done = int(next_done)
        if self.env.done:
          self.logger.info(f"pid={os.getpid()} End episode: gloabl_step={self.global_step}, episodic_reward={self.env.total_reward:.2f}, " +
                   f"episodic_score={int(self.env.total_score)}, " +
                   f"miss_treasure={self.env.miss_treasure}, n_treasure={self.env.n_treasure}")
          self.writer.add_scalar("charts/episodic_reward", self.env.total_reward, self.global_step)
          self.writer.add_scalar("charts/episodic_score", self.env.total_score, self.global_step)
          self.writer.add_scalar("charts/episodic_length", self.env.n_step, self.global_step)
          self.writer.add_scalar("charts/hit_wall", self.env.total_hit_wall, self.global_step)
          self.writer.add_scalar("charts/miss_treasure", self.env.miss_treasure, self.global_step)
          self.writer.add_scalar("charts/n_treasure", self.env.n_treasure, self.global_step)
          # self.monitor.put_data({os.getpid(): {
          #   'episodic_reward': self.env.total_reward,
          #   'episodic_score': self.env.total_score,
          #   'episodic_length': self.env.n_step,
          #   'hit_wall': self.env.total_hit_wall,
          #   'miss_treasure': self.env.miss_treasure,
          #   'n_treasure': self.env.n_treasure,
          # }})

      # bootstrap value if not done, 用GAE计算优势函数
      with torch.no_grad():
        next_value = self.model.get_value(torch.Tensor(next_obs).to(self.device).view(1, -1)).cpu().numpy()[0,0]
      lastgaelam = 0
      for t in reversed(range(args.num_steps)):
        if t == args.num_steps - 1:
          nextnonterminal = 1.0 - next_done
          nextvalues = next_value
        else:
          nextnonterminal = 1.0 - dones[t + 1]
          nextvalues = values[t + 1]
        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        collector[t].advantage = lastgaelam
        collector[t].ret = lastgaelam + values[t]

      self.learn(sample_process(collector))  # upload collector to buffer
      SPS = int(self.global_step / (time.time() - self.start_time))
      time_left = (args.num_iterations * args.num_steps - self.global_step) / SPS
      self.writer.add_scalar("time/SPS", SPS, self.global_step)
      self.writer.add_scalar("time/time_left", time_left, self.global_step)
      # self.try_save_model(120)

  @save_model_wrapper
  def save_model(self, path=None, id="1"):
    model_file_path = f"{path}/model.ckpt-{str(id)}.pth"
    model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
    torch.save(model_state_dict_cpu, model_file_path)
    self.logger.info(f"save model {model_file_path} successfully")

  @load_model_wrapper
  def load_model(self, path=None, id="1"):
    model_file_path = f"{path}/model.ckpt-{str(id)}.pth"
    self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
    self.logger.info(f"pid={os.getpid()} load_model_path={model_file_path}")
  
  def try_save_model(self, delta=120):
    now = time.time()
    if now - self.last_opt_time > delta:
      self.save_model()
      self.last_opt_time = now
