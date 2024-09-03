"""
用于绘制当前智能体位置
map_1.json: 来自 kaiwu_env/conf/back_to_the_realm/map_data/map_1.json
crab.yaml: 来自 kaiwu_env/conf/back_to_the_realm/treasure_path/crab.yaml

图中编号分别为:
0: 中间加速buff
1: 起点 (测试时用)
2: 终点 (测试时用)
3-15: 宝箱位
"""
import numpy as np
import cv2
import json
import yaml
from pathlib import Path

PATH_ROOT = Path(r"/data/projects/back_to_the_realm")
PATH_FIGURE = PATH_ROOT / 'figures'
PATH_FIGURE.mkdir(exist_ok=True, parents=True)
PATH_PARENT = Path(__file__).parent
path_grid = PATH_PARENT / r"data/map_1.json"
path_treasure = PATH_PARENT / r"data/crab.yaml"
COLORS = {
  'wall': (255, 255, 255),
  'treasure': (255, 100, 0),
  'text': (0, 255, 0),
  'agent': (0, 0, 255),
}

class Drawer:
  def __init__(self, path_grid=path_grid, path_treasure=path_treasure):
    with open(path_grid, 'r') as file:
      d = json.load(file)
    self.cell_width = d['CellWidth']
    self.w, self.h = d['Width'], d['Height']
    self.size = np.array((self.h, self.w), np.int32)
    self.grid = np.array(d['Flags'], bool).reshape(self.h, self.w)
    self.img = np.zeros((self.h, self.w, 3), np.uint8)

    self.treasures = []
    if path_treasure is not None:
      with open(path_treasure, 'r') as file:
        d = yaml.load(file.read(), yaml.FullLoader)
      for i, pos in d.items():
        pos = np.array(pos, np.int32) // self.cell_width
        self.treasures.append(pos)
      self.treasures = np.stack(self.treasures)
  
  def draw_map(self, flags=None):
    if flags is not None:
      assert len(flags) == len(self.treasures)
    img = np.zeros_like(self.img)
    img[self.grid] = COLORS['wall']
    img = cv2.flip(img, 0)
    if len(self.treasures):
      for i, pos in enumerate(self.treasures):
        if flags is not None and not flags[i]: continue
        pos = pos + np.array([5, 10])
        pos[1] = self.h - pos[1]
        img[pos[1], pos[0]] = COLORS['treasure']
        # print(type(img), img.dtype, img.shape)
        img = cv2.putText(img, str(i), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text'], 1)
    self.img = img
  
  def savefig(self, filename='map'):
    cv2.imwrite(str(PATH_FIGURE/f"{filename}.png"), self.img)
  
  def draw_agent(self, y, x):
    self.img[self.h - y, x] = COLORS['agent']
  
  def drawflow(self, y, x, flags=None, filename='map'):
    self.draw_map(flags)
    self.draw_agent(y, x)
    self.savefig(filename)
  
  def draw_local_map(self, map: np.ndarray, filename='local_map'):
    if map.ndim == 2:
      map = np.stack([map]*3, -1)
    map = (map * 255).astype(np.uint8)
    map[25, 25] = COLORS['agent']
    cv2.imwrite(str(PATH_FIGURE/f"{filename}.png"), map)

if __name__ == '__main__':
  grid_drawer = Drawer(path_grid, path_treasure)
  flags = None
  # flags = np.zeros(16, bool)
  # flags[:6] = True
  grid_drawer.drawflow(30, 10)
