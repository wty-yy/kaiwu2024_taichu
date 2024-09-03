import numpy as np

class Config:

  STATE_SIZE = 64 * 64 * 1024
  ACTION_SIZE = 4
  LEARNING_RATE = 0.8
  GAMMA = 0.9
  EPSILON = 0.1
  EPISODES = 10000

  # dimensionality of the sample
  # 样本维度
  SAMPLE_DIM = 5

  # Dimension of movement action direction
  # 移动动作方向的维度
  OBSERVATION_SHAPE = 214

  # The following configurations can be ignored
  # 以下是可以忽略的配置
  LEGAL_ACTION_SHAPE = 0
  SUB_ACTION_MASK_SHAPE = 0
  LSTM_HIDDEN_SHAPE = 0
  LSTM_CELL_SHAPE = 0

  DIM_OF_ACTION = 4
  
  ### Environment config ###
  # rotate 90 deg clockwise
  # up, down, left, right -> right, left, up, down
  dpos = np.array([(0, 1), (0, -1), (-1, 0), (1, 0)], np.int32)
  key2idx = {'w': 2, 's': 3, 'a': 0, 'd': 1}
  ENV_CFG = {
    'diy': {
      'start': [29, 9],  # Start position
      'end': [11, 55],  # End position
      'treasure_id': [0, 1, 2, 3, 4],
      # 'treasure_id': [0,1,2,3,4,5,6,7,8,9],  # Specify treasure indexes (treasure_random==0)
      # 'treasure_num': 5,  # Generate treasure number (treasure_random==1)
      # 'treasure_random': 0,
    }
  }
  targets = [
    (19, 14), (23, 23), (34, 23), (41, 33), (44, 30), (54, 41), (49, 56),  # 0, 4, 8, 9, 5
    (54, 41), (44, 30), (38, 40), (42, 45), (22, 40), (9, 40), (9, 44), (9, 28),  # 3, 2, 1
    (9, 40), (22, 40), (23, 55), (35, 58), (23, 55), (11, 55)  # 7, 6
  ]
  # targets = [
  #   (30, 14)
  # ]

Config = Config()