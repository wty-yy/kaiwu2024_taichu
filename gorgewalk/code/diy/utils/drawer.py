"""
Maze drawer, save figure `maze.png` in current file directory.
"""
import cv2
import numpy as np
from diy.utils.keyboard import wait_key
import time
from pathlib import Path
from diy.utils import show_debug
# PATH_ROOT = Path(__file__).parents[1]
# PATH_LOG = PATH_ROOT / "log"  # Autoremove, so bad...

COLORS = {  # bgr
  'red': (0, 0, 255),
  'green': (0, 128, 0),
  'blue': (255, 0, 0),
  'yellow': (0, 255, 255),
  'orange': (0, 165, 255),
  'white': (255, 255, 255),
}
cfg = {
  'color': {
    'memory': COLORS['green'],
    'star': COLORS['yellow'],
    'now': COLORS['red'],
    'start': COLORS['orange'],
    'end': COLORS['orange'],
    'free': COLORS['white'],
    'wall': COLORS['blue']
  }
}

dpos = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)], np.int32)
key2idx = {'w': 0, 's': 1, 'a': 2, 'd': 3}

def pos2idx(pos):
  return pos[0] * 64 + pos[1]

def idx2pos(idx):
  return np.array((idx // 64, idx % 64), np.int32)

def state2dict(s):
  assert len(s) == 250
  s = np.array(s, np.float32)
  s = {
    'pos': (np.argmax(s[1:65]), np.argmax(s[65:129])),
    'dist': {
      'end': s[129],
      'star': s[130:140],  # should use star_flag to check the star existed, if star is get
    },
    'map': {
      'wall': s[140:165].reshape(5, 5),
      'star': s[165:190].reshape(5, 5),
      'end': s[190:215].reshape(5, 5),
    },
    'memory': s[215:240].reshape(5, 5),
    'star_flag': s[240:250]
  }
  types = {
    'pos': np.int32,
    'dist': np.int32,
    'map': bool,
    'memory': np.float32,
    'star_flag': bool 
  }
  for k, v in s.items():
    t = types[k]
    if not isinstance(v, dict):
      s[k] = np.array(v, t)
    else:
      for kk in v:
        v[kk] = np.array(v[kk], types[k])
  return s

class Drawer:
  def __init__(self, file_name='maze'):
    self.img = np.zeros((64, 64, 3), np.uint8)
    # self.path_save = str(PATH_LOG / (file_name + '.png'))
    self.path_save = str(Path(__file__).parent / (file_name + '.png'))
    self.build_order = ['free', 'wall', 'memory', 'now', 'star', 'start', 'end']
    self.free = set()  # white
    self.wall = set()  # blue
    self.memory = set()  # green
    self.now = None  # red
    self.star = set()  # yellow
    self.start = None  # orange
    self.end = None  # orange

  def update_now(self, now):
    if self.start is None: self.start = np.array(now, np.int32)
    self.now = np.array(now, np.int32)
    self.memory.add(pos2idx(now))
  
  def update_relative(self, map: np.ndarray, is_wall=False, is_star=False, is_end=False):
    """
    Args:
      map: [np.ndarray, shape=(5,5), type=bool] Around boolean map relative to self.now
      color0: [str | tuple] Color for False position in map
      color0: [str | tuple] Color for True position in map
    """
    for i in range(map.shape[0]):
      for j in range(map.shape[1]):
        pos = self.now + np.array([i-2, j-2], np.int32)
        idx = pos2idx(pos)
        if map[i,j]:
          if is_end:
            self.end = pos.copy()
          for flag, storage in zip(
            (is_wall, is_star),
            (self.wall, self.star)):
            if flag:
              storage.add(idx)
        if not map[i,j] and is_wall:
          self.free.add(idx)
  
  def build(self, save=True):
    ccfg = cfg['color']
    self.img = np.zeros_like(self.img)
    for key in self.build_order:
      color = ccfg[key]
      x = getattr(self, key)
      if x is None: continue
      if isinstance(x, set):
        for idx in x:
          self.draw(idx2pos(idx), color)
      else:
        self.draw(x, color)
    if save: self.save()

  def draw(self, pos, color):
    self.img[pos[0], pos[1]] = color
  
  def save(self, counter_clockwise=True):
    if counter_clockwise:
      img = cv2.flip(cv2.transpose(self.img), 0)
    cv2.imwrite(self.path_save, img)
    
    # show_debug(f"save maze figure in {self.path_save}")
  
  def update_state(self, state: dict, raw_state: bool = True):
    """
    Args:
      state: [dict] Defined in train_wrokflow.py
    """
    if raw_state:
      state = state2dict(state)
    self.update_now(state['pos'])
    map = state['map']
    self.update_relative(map['wall'], is_wall=True)
    self.update_relative(map['star'], is_star=True)
    self.update_relative(map['end'], is_end=True)

if __name__ == '__main__':
  drawer = Drawer()
  now = np.array((5, 10), np.int32)
  drawer.update_now(now)
  drawer.build()
  while True:
    x = wait_key()
    if x == chr(27): break  # ESC
    if x not in key2idx:
      print(f"Don't know key '{x}'")
      continue
    delta = dpos[key2idx[x]]
    now += delta
    drawer.update_now(now)
    drawer.build()
