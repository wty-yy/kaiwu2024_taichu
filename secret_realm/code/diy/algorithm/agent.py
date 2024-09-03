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
from diy.utils.ckpt_manager import clean_ckpt_memory, get_latest_ckpt_path
PATH_ROOT = Path(__file__).parents[2]
PATH_LOGS_DIR = PATH_ROOT / "log/tensorboard"
PATH_LOGS_DIR.mkdir(exist_ok=True, parents=True)

def init_writer(agent_type):
    run_name = f"secret_realm_ppo_v{args.version}_{agent_type}_pid{os.getpid()}_{time.strftime(r'%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(str(PATH_LOGS_DIR / run_name))
    return writer

class Agent(BaseAgent):

  def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
    super().__init__(agent_type, device, logger, monitor)
    self.logger, self.monitor = logger, monitor
    logger.info(f"pid={os.getpid()}, {device=}")
    self.device = device
    self.model = Model().to(self.device)
    self.env: SecretRealmEnv = None

    self.global_step = 0
    if agent_type == 'learner':
      self.start_time = None
      self.writer = init_writer('learner')
      self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, eps=1e-5)
    if args.load_model_id is not None:
      self.logger.info("Load pre-trained model.")
      self._load_model(path="/data/projects/back_to_the_realm/ckpt", id=args.load_model_id)

  def _predict(self, list_obs_data):
    obs = list_obs_data[0].feature
    obs = torch.Tensor(obs).to(self.device).view(1, -1)
    with torch.no_grad():
      action, logprob, _, _ = self.model.get_action_and_value(obs)
    return [ActData(
      act=int(action[0].cpu().numpy()),
      logprob=logprob[0].cpu().numpy()
    )]

  @predict_wrapper
  def predict(self, list_obs_data):
    return self._predict(list_obs_data)

  @exploit_wrapper
  def exploit(self, list_obs_data):
    return self._predict(list_obs_data)

  @learn_wrapper
  def learn(self, trajs):
    """
    Args:
      one_traj: [List[SampleData]] trajectory from aisrv
    """
    if self.start_time is None:
      self.logger.info(f"n_sample={len(trajs)}")
      self.start_time = self.last_opt_time = time.time()
    num_sample = len(trajs)  # train_batch_size in configure_app.toml
    num_steps = len(trajs[0].obs)
    assert num_steps == args.num_steps
    learn_start_time = time.time()
    args.batch_size = num_sample * args.num_steps
    # args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_minibatches = args.batch_size // args.minibatch_size

    list2tensor = lambda x, axis: torch.tensor(np.stack(x, axis)).to(self.device)
    obs = list2tensor([x.obs for x in trajs], 1)  # (num_steps, num_sample, obs_dim)
    actions = list2tensor([x.actions for x in trajs], 1)  # (num_steps, num_sample)
    rewards = list2tensor([x.rewards for x in trajs], 1)  # (num_steps, num_sample)
    dones = list2tensor([x.dones for x in trajs], 1)  # (num_steps, num_sample)
    next_obs = list2tensor([x.next_obs for x in trajs], 0)  # (num_sample, obs_dim)
    next_done = list2tensor([x.next_done for x in trajs], 0)  # (num_sample,)
    logprobs = list2tensor([x.logprobs for x in trajs], 1)  # (num_steps, num_sample)
    # print(f"{obs.shape=}, {actions.shape=}, {rewards.shape=}, {dones.shape=}, {next_obs.shape=}, {next_done.shape=}")

    # Action logic
    with torch.no_grad():
      _, _, _, values = self.model.get_action_and_value(obs.view(-1, args.obs_dim), actions.long().view(-1))
    values = values.view(num_steps, num_sample)
    
    # Calculate GAE
    with torch.no_grad():
      next_value = self.model.get_value(next_obs).flatten()  # (num_sample,)
    lastgaelam = 0
    advantages = torch.zeros_like(rewards).to(self.device)
    for t in reversed(range(num_steps)):
      if t == args.num_steps - 1:
        nextnonterminal = 1.0 - next_done
        nextvalues = next_value
      else:
        nextnonterminal = 1.0 - dones[t + 1]
        nextvalues = values[t + 1]
      delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
      advantages[t] = lastgaelam = (
        delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam)
    returns = advantages + values  # (num_steps, num_sample)

    # Flatten the batch
    b_obs = obs.view(-1, args.obs_dim)
    b_logprobs = logprobs.view(-1)
    b_actions = actions.view(-1)
    b_advantages = advantages.view(-1)
    b_returns = returns.view(-1)
    b_values = values.view(-1)

    # Optimizing the policy and value network
    self.logger.info(f"{args.batch_size=}, {args.minibatch_size=}")
    b_inds = np.arange(args.batch_size)
    clipfracs, grad_norms = [], []
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
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
        grad_norms.append(float(grad_norm))
        self.optimizer.step()
        self.global_step += 1

      if args.target_kl is not None and approx_kl > args.target_kl:
        break

    # 计算方差解释率，衡量模型预测value的准确性
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
    self.writer.add_scalar("losses/grad_norm", np.mean(grad_norms), self.global_step)
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
    if now - self.last_opt_time > 120:
      self.last_opt_time = now
      self.save_model()  # save to local computer
      clean_ckpt_memory(path_name='restore', logger=self.logger)

  @save_model_wrapper
  def save_model(self, path=None, id="1"):
    model_file_path = f"{path}/model.ckpt-{str(id)}.pth"
    model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
    torch.save(model_state_dict_cpu, model_file_path)
    self.logger.info(f"save model {model_file_path} successfully")

  @load_model_wrapper
  def load_model(self, path=None, id="1"):
    self._load_model(path, id)
  
  def _load_model(self, path, id):
    try:
      model_file_path = f"{path}/model.ckpt-{str(id)}.pth"
      self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
      self.logger.info(f"pid={os.getpid()} load_model_path={model_file_path}")
    except FileNotFoundError:
      self.logger.info(f"File {model_file_path} not found.")
  
  def try_save_model(self, delta=120):
    now = time.time()
    if now - self.last_opt_time > delta:
      self.save_model()
      self.last_opt_time = now
