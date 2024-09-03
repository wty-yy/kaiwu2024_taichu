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
from diy.config import Config
from diy.utils import show_debug


@attached
class Agent(BaseAgent):
  def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
    super().__init__(agent_type, device, logger, monitor)
    self.logger = logger
    self.verbose = False
    # show_debug('init Agent', verbose_depth=100)
    self.reset()
  
  def reset(self):
    if self.verbose:
      self.drawer = Drawer()
    self.ti, self.target, self.inv = 0, Config.targets[0], False
    self.last_pos = None
    self.n_step = 0

  def _predict(self, list_obs_data):
    self.n_step += 1
    s = list_obs_data[0].feature
    if self.verbose:
      self.drawer.update_state(s)
      self.drawer.build()
    pos = s['pos']
    if self.last_pos is not None and np.all(pos == self.last_pos):
      self.inv ^= 1
    self.last_pos = pos.copy()
    delta = np.array(self.target, np.int32) - pos
    # show_debug(f"{self.target=}, {pos=}, {delta=}, {s['pos']=}, {self.inv=}, {self.n_step=}, {self.ti=}, {s['memory']=}, {s['dist']['star']=}")
    # if s['map']['star'][2,2]:
    #   show_debug(f"get star")
    # self.logger.info(f"{self.target=}, {pos=}, {delta=}, {s['pos']=}, {self.inv=}, {self.n_step=}, {self.ti=}")
    self.logger.info(f"{s['dist']['end']=}, {s['dist']['star']=}, {s['star_flag']=}")
    if not np.any(delta):
      self.ti += 1
      self.target = Config.targets[self.ti]
      delta = np.array(self.target, np.int32) - pos
    for a, d in enumerate(Config.dpos[::-1] if self.inv else Config.dpos):
      if sum(d * delta) > 0:
        break
    if self.inv: a = 3 - a
    return [ActData(act=a)]

  @exploit_wrapper
  def exploit(self, list_obs_data):
    return self._predict(list_obs_data)

  @predict_wrapper
  def predict(self, list_obs_data):
    return self._predict(list_obs_data)
    # return [ActData(act=0)]

  @learn_wrapper
  def learn(self, list_sample_data):
    # show_debug('agent.learn', verbose_depth=100)
    pass

  @save_model_wrapper
  def save_model(self, path=None, id="1"):
    show_debug(f"Save model {path=} {id=}", verbose_depth=100)
    import threading
    show_debug(f"Threading:" + str(threading.enumerate()))
    path = f"{path}/model.ckpt-{str(id)}.npy"
    np.save(path, {})

  @load_model_wrapper
  def load_model(self, path=None, id="1"):
    pass
