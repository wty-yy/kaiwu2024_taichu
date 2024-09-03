import numpy as np
from kaiwu_agent.agent.base_agent import (
  BaseAgent,
  predict_wrapper,
  exploit_wrapper,
  learn_wrapper,
  save_model_wrapper,
  load_model_wrapper,
)
from kaiwu_agent.utils.common_func import attached
from diy.feature.definition import ActData
from diy.utils.drawer import Drawer
from diy.utils import show_debug


@attached
class Agent(BaseAgent):
  def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
    super().__init__(agent_type, device, logger, monitor)
    self.logger = logger
    # show_debug("Init Agent verbose_depth=100)
  
  def reset(self):
    if self.verbose:
      self.drawer = Drawer()

  @exploit_wrapper
  def exploit(self, list_obs_data):
    return [ActData(act=[0 for _ in range(16)])]

  @predict_wrapper
  def predict(self, list_obs_data):
    return [ActData(act=[0 for _ in range(16)])]

  @learn_wrapper
  def learn(self, list_sample_data):
    pass

  @save_model_wrapper
  def save_model(self, path=None, id="1"):
    show_debug(f"Save model {path=} {id=}")
    path = f"{path}/model.ckpt-{str(id)}.npy"
    np.save(path, {})

  @load_model_wrapper
  def load_model(self, path=None, id="1"):
    pass
