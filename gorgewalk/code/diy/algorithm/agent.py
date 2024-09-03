import time
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
from diy.feature.definition import ActData, GorgeWalk
from diy.model.model import Model
from diy.utils import show_time, show_debug
from pathlib import Path
PATH_ROOT = Path(__file__).parents[2]
PATH_LOGS_DIR = PATH_ROOT / "runs"
PATH_LOGS_DIR.mkdir(exist_ok=True)

class Agent(BaseAgent):

    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        super().__init__(agent_type, device, logger, monitor)
        self.logger, self.moniter = logger, monitor
        self.device = torch.device("cpu")
        self.model = Model().to(self.device)
        self.args = None
        self.envs: GorgeWalk = None

    def get_everything(self, args, envs):
        self.args = args
        self.envs = envs

    @predict_wrapper
    def predict(self, list_obs_data):
        return [ActData(act=1)]

    @exploit_wrapper
    def exploit(self, list_obs_data):
        obs = list_obs_data[0].feature
        obs = torch.Tensor(obs).to(self.device).view(1, -1)
        action, _, _, _ = self.model.get_action_and_value(obs)
        return [ActData(act=int(action[0].cpu().numpy()))]

    def train(self):
        assert self.model is not None, "model is None"
        assert self.args is not None, "args is None"
        assert self.envs is not None, "envs is None"
        args = self.args
        args.batch_size = int(args.num_envs * args.num_steps) # 128
        args.minibatch_size = int(args.batch_size // args.num_minibatches) # 32
        args.num_iterations = args.total_timesteps // args.batch_size # 23
        run_name = f"gorge_walk_ppo_v{args.version}__{time.strftime('%Y%m%d-%H%M%S')}"

        try:
            import wandb
            wandb.init(
                project="kaiwu2024_gorge_walk",
                entity="vainglory",  # commit this, if you don't need entity
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
            )
        except ModuleNotFoundError:
            self.logger.info("No wandb package, skip upload tensorboard.")

        writer = SummaryWriter(str(PATH_LOGS_DIR / run_name))
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, eps=1e-5)
        obs = torch.zeros((self.args.num_steps, self.args.num_envs) + self.envs.observation_space).to(self.device)
        actions = torch.zeros((self.args.num_steps, self.args.num_envs) + self.envs.action_space).to(self.device)
        logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        global_step = 0
        start_time = time.time()
        next_obs, _ = self.envs.reset()
        next_obs = torch.Tensor(next_obs).to(self.device).view(1,-1)
        next_done = torch.zeros(self.args.num_envs).to(self.device)
        for iteration in range(1, self.args.num_iterations + 1):
            frac = (iteration - 1.0) / self.args.num_iterations
            # Adjust learning rate
            if self.args.anneal_lr:  # Annealing the rate if instructed to do so.
                lrnow = (1.0 - frac) * self.args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow
            elif len(self.args.group_lr): # Adjust learning rate by proportion
                idx = int(frac * len(self.args.group_lr))
                optimizer.param_groups[0]['lr'] = self.args.group_lr[idx]
            # Adjust entropy coefficient
            if len(self.args.group_ent_coef):
                idx = int(frac * len(self.args.group_ent_coef))
                self.args.ent_coef = self.args.group_ent_coef[idx]

            for step in range(0, self.args.num_steps):
                global_step += self.args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.model.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = self.envs.step(action[0].cpu().numpy())
                # print("wall", next_obs[139:164].reshape(5, 5))
                # print("reward:", reward)

                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs= torch.Tensor(next_obs).to(self.device).view(1,-1)
                next_done = torch.Tensor([1]).to(self.device) if next_done else torch.Tensor([0]).to(self.device)
                if self.envs._done:
                    self.logger.info(f"End episode: global_step={global_step}, episodic_reward={self.envs.total_reward:.2f}, " +
                                     f"episodic_score={int(self.envs.total_score)}, miss_treasure={int(self.envs.exist_treasure)}, " +
                                     f"n_treasure={self.envs.n_treasure}")
                    writer.add_scalar("charts/episodic_reward", self.envs.total_reward, global_step)
                    writer.add_scalar("charts/episodic_score", self.envs.total_score, global_step)
                    writer.add_scalar("charts/episodic_length", self.envs.total_length, global_step)
                    writer.add_scalar("charts/hit_wall", self.envs.hit_wall, global_step)
                    writer.add_scalar("charts/miss_treasure", self.envs.exist_treasure, global_step)
                    writer.add_scalar("charts/n_treasure", self.envs.n_treasure, global_step)

            # bootstrap value if not done, 用GAE计算优势函数
            with torch.no_grad():
                next_value = self.model.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.args.num_steps)):
                    if t == self.args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + self.args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[
                        t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1, ) + self.envs.observation_space)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1, ) + ())
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.args.batch_size)
            clipfracs = []
            for epoch in range(self.args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.args.batch_size, self.args.minibatch_size):
                    end = start + self.args.minibatch_size
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
                        clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() +
                                                                                1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef,
                                                            1 + self.args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds])**2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.args.clip_coef,
                            self.args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds])**2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds])**2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    self.learn([None])

                if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                    break

            # 计算方差解释率，衡量模型预测的准确性
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            
            
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            SPS = int(global_step / (time.time() - start_time))
            time_left = (args.num_iterations * args.num_steps - global_step) / SPS
            self.logger.info(f"SPS: {SPS}, " + f"time left: {show_time(time_left)}")
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # envs.close()
        writer.close()

    @learn_wrapper
    def learn(self, list_sample_data):
        pass

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # ...
        model_file_path = f"{path}/model.ckpt-{str(id)}.pth"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pth"
        self.model.load_state_dict(torch.load(model_file_path))
