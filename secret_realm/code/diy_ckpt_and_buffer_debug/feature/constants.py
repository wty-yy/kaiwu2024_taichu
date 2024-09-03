import numpy as np

delta_pos = np.array([
  [0, 1], [1, 1], [1, 0], [1, -1],
  [0, -1], [-1, -1], [-1, 0], [-1, 1]
], np.int32)

delta_dist_walk = 600  # max 711
delta_dist_buff = 900 # 996
delta_dist_flash = 7900  # 7996