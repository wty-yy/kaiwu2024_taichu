from kaiwu_agent.utils.common_func import create_cls, attached
import numpy as np
from diy.config import Args
import json, time
from diy.utils import show_iter
from diy.utils.drawer import Drawer
import diy.feature.constants as const

SampleData = create_cls("SampleData", a=None)
ObsData = create_cls("ObsData", feature=None)
ActData = create_cls("ActData", act=None)

class SecretRealmEnv:
  def __init__(self, env, logger):
    self.env = env
    self.logger = logger
    self.observation_space = (Args.obs_dim,)
    self.action_space = ()
    self.episodes = self.total_timestep = 0
    self.treasure_miss_cnt = np.zeros((10,), np.int32)
    self.drawer = Drawer()
  
  def reset(self):
    self.usr_conf = {
      'diy': {
        'start': 2,  # range in [1, 15]
        'end': 1,  # range in [1, 15], diff with start
        'treasure_random': 0,  # if toggled, choosing treasure positions randomly
        # 'treasure_num': 3,
        'treasure_id': [3, 7, 15],  # if treasure_random==1, treasures are generate based on this
        'max_step': 2000,
        'talent_type': 1,  # can't change
      }
    }
    obs = self.env.reset(self.usr_conf)
    while obs is None:
      obs = self.env.reset(self.usr_conf)
    obs = observation_process(obs).feature  # obs2dict
    self.n_step = self.total_reward = self.total_score = \
      self.total_length = self.n_hit_wall = 0
    self.done = False
    # DEBUG
    with open("/data/projects/back_to_the_realm/obs.yaml", 'w') as file:
      file.write(show_iter(obs))
    self.drawer.drawflow(*obs['grid_pos'])
    self._obs = obs
    return obs
  
  def step(self, action):
    """
    Args: 
      action: [np.ndarray, shape=(2,)]
        action[0] is direction in 0~7, action[1] is whether using flash
    """
    self.action = action[0] + action[1] * 8
    # time1 = time.time()
    frame_no, obs, score, terminated, truncated, _env_info = self.env.step(self.action)
    print("[DEBUG] score: ", int(score.score))
    self.obs = obs = observation_process(obs).feature
    self.done = terminated | truncated
    self.n_step += 1
    self.info = info = info2dict(_env_info)
    self.hit_wall = self._check_hit_wall()
    # DEBUG
    # if self.hit_wall:
    #   print(f"{self.n_step=}: hit_wall")
    # if self.n_step % 1 == 0:
    # with open(f"/data/projects/back_to_the_realm/runs/obs{self.n_step}.yaml", 'w') as file:
    #   file.write(show_iter(obs))
    # with open(f"/data/projects/back_to_the_realm/runs/info{self.n_step}.yaml", 'w') as file:
    #   file.write(show_iter(info))
    # self.drawer.drawflow(*obs['grid_pos'], filename=f"map{self.n_step}")
    # self.drawer.draw_local_map(obs['obstacle_map'], f"obstacle_map{self.n_step}")
    # self.drawer.draw_local_map(obs['treasure_map'], f"treasure_map{self.n_step}")
    self._obs = obs
    # print(f"{self.n_step=}, {obs['memory_map'][22:28,22:28]=}")

    ### Reward ###
    reward = score
    return obs, reward, self.done, info
  
  def _check_hit_wall(self):
    if not self._check_has_wall_around(): return False
    on_buff = self.info['buff_remain_time'] > 0
    use_flash = self._obs['legal_act'][1] - self.obs['legal_act'][1]
    _abs_pos, abs_pos = [x['norm_pos'] * 64000 for x in [self._obs, self.obs]]
    dist = np.linalg.norm(_abs_pos - abs_pos)
    if (not on_buff and dist < const.delta_dist_walk or
        on_buff and dist < const.delta_dist_buff or
        use_flash == 1 and dist < const.delta_dist_flash):
      return True
    return False
  
  def _check_has_wall_around(self, alpha=2):
    """
    Check whether there a wall within the distance alpha relative to 
    last position. (only consider action direction vector)
    """
    wall_map = self._obs['obstacle_map']  # 0: wall, 1: no-wall
    x = np.array([25, 25], np.int32)
    for i in range(-1, 2):
      action = (self.action + i + 8) % 8
      dx = const.delta_pos[action % 8]
      for a in range(alpha):
        tx = x + dx * a
        if not wall_map[tx[0], tx[1]]: return True
    # print(f"{self.n_step=}", tx)
    # print(wall_map[23:28,23:28])
    return False

def relative2dict(relative_position):
  """
  Convert Relative Position to Dict

  If object doesn't exist (non-generated or gained):
    direction=0, l2_distance=path_distance=5, grid_distance=1.0
  """
  return {
    'direction': relative_position.direction,  # 0 is None, 1~8 is direction (object relative to hero direction)
    'l2_distance': relative_position.l2_distance,  # discrete straight line l2 distance, in [1, 5]
    ## BFS distance ###
    'path_distance': relative_position.path_distance,  # discrete distance, in [1, 5]
    'grid_distance': relative_position.grid_distance,  # normal distance, in (0, 1)
  }
    
def obs2dict(raw_obs):
  obs = {
    # in (0, 1) = absolute coord / 64000, (64000 = 128 * 500), float
    'norm_pos': np.array((raw_obs.feature.norm_pos.x, raw_obs.feature.norm_pos.z), np.float32),
    # in (128, 128), used for drawing, int
    'grid_pos': np.array((raw_obs.feature.grid_pos.x, raw_obs.feature.grid_pos.z), np.int32),
    ### Relative Position ###
    'start_pos': relative2dict(raw_obs.feature.start_pos),
    'end_pos': relative2dict(raw_obs.feature.end_pos),
    'buff_pos': relative2dict(raw_obs.feature.buff_pos),
    'treasure_pos': [relative2dict(pos) for pos in raw_obs.feature.treasure_pos],
    ### Map Figure 51x51 ###
    'obstacle_map': np.array(raw_obs.feature.obstacle_map, np.int32).reshape(51, 51),
    'memory_map': np.array(raw_obs.feature.memory_map, np.float64).reshape(51, 51),
    'treasure_map': np.array(raw_obs.feature.treasure_map, np.int32).reshape(51, 51),
    'end_map': np.array(raw_obs.feature.end_map, np.int32).reshape(51, 51),
    # Mask for available action, shape=(2,) always legal_act[0]=1, if legal_act[1]=1 skill is available
    'legal_act': np.array(raw_obs.legal_act, np.int32),
  }
  return obs

def info2dict(env_info):
  frame_state = env_info.frame_state
  hero = frame_state.heroes[0]
  talent = hero.talent
  game_info = env_info.game_info
  info = {
    ### FrameState ###
    'n_frame': frame_state.frame_no,  # frame number, start from 1, add 3 at each step
    'hero': {
      'id': hero.hero_id,
      'pos': np.array((hero.pos.x, hero.pos.z), np.int32),
      'speed_up': bool(hero.speed_up),  # whether hero speed up
      'talent': {  # hero skill
        'type': talent.talent_type,
        'status': bool(talent.status),  # whether skill is available
        'cooldown': talent.cooldown,  # cooldown time of skill
      }
    },
    'treasures': [],  # all treasures state
    'buff': {},  # one buff state
    ### GameInfo ###
    'score': game_info.score,  # total treasure score
    'total_score': game_info.total_score,  # total score (treasure score + step score + end score)
    'n_step': game_info.step_no,  # step number
    'hero_pos': np.array((game_info.pos.x, game_info.pos.z), np.int32),  # hero position
    'treasure_count': game_info.treasure_count,  # number of treasures obtained
    'treasure_score': game_info.treasure_score,  # score of treasures obtained
    'buff_count': game_info.buff_count,  # number of buff obtained
    'talent_count': game_info.talent_count,  # number of talent used
    'buff_remain_time': game_info.buff_remain_time,  # remain time of the buff
    'buff_duration': game_info.buff_duration,  # duration time of the buff
  }
  organs = frame_state.organs
  for organ in organs:
    tmp = {
      'id': organ.config_id,
      'status': bool(organ.status),  # whether the object is exists
      'pos': np.array((organ.pos.x, organ.pos.z), np.int32),  # always be [0, 0]
      'cooldown': organ.cooldown,  # cooldown time, useless (treasure cooldown must be 0, buff won't be reset)
    }
    if organ.sub_type == 1:  # treasure
      info['treasures'].append(tmp)
    else:  # buff
      info['buff'] = tmp
  return info

@attached
def observation_process(raw_obs):
  return ObsData(feature=obs2dict(raw_obs))

@attached
def action_process(act_data):
    return act_data.act

@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]
