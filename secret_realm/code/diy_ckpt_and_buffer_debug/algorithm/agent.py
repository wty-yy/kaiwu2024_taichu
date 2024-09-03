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
from diy.utils.ckpt_manager import clean_ckpt_memory, get_latest_ckpt_path
from kaiwudrl.common.config.config_control import CONFIG
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
    self.count = 0
    self.last_opt_time = 0
    self.model_update_times = 0

  @predict_wrapper
  def predict(self, list_obs_data):
    return [ActData(act=0)]

  @exploit_wrapper
  def exploit(self, list_obs_data):
    return [ActData(act=0)]

  @learn_wrapper
  def learn(self, batch_data):
    # self.logger.info(f"{batch_data=}")
    info = ""
    for i, data in enumerate(batch_data):
      info += f"{i=}, {data.a=}\n"
    self.logger.info(info)
    self.model_update_times += 1
    now = time.time()
    if now - self.last_opt_time > 10:
      clean_ckpt_memory()
      self.last_opt_time = now

  def rollout(self):
    for _ in range(100000):
      frame = Frame(a=self.count)
      self.learn(sample_process([frame]))
      time.sleep(0.1)
      self.count += 1
      now = time.time()
      if now - self.last_opt_time > 10:
        # self.load_model(id='latest')
        self.last_opt_time = now
        path = get_latest_ckpt_path()
        if path is None:
          self.logger.info(f"There is no ckpt in {CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/")
        self.myload_model(path)

  @save_model_wrapper
  def save_model(self, path=None, id="1"):
    model_file_path = f"{path}/model.ckpt-{str(id)}.npy"
    np.save(model_file_path, np.array(self.model_update_times, np.int32))
    self.logger.info(f"save model {model_file_path} successfully")

  @load_model_wrapper
  def load_model(self, path=None, id="1"):
    model_file_path = f"{path}/model.ckpt-{str(id)}.npy"
    a = np.load(model_file_path)
    self.logger.info(f"pid={os.getpid()} load_model_path={model_file_path}, load={a}")
  
  def myload_model(self, path):
    a = np.load(path)
    self.logger.info(f"pid={os.getpid()} load_model_path={path}, load={a}")
  
  def try_save_model(self, delta=120):
    now = time.time()
    if now - self.last_opt_time > delta:
      self.save_model()
      self.last_opt_time = now
