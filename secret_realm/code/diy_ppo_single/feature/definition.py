from kaiwu_agent.utils.common_func import create_cls, attached
import numpy as np
from diy.config import args
import json, time
from diy.utils import show_iter
import diy.feature.constants as const

SampleData = create_cls("SampleData", image=None)
ObsData = create_cls("ObsData", feature=None)
ActData = create_cls("ActData", act=None)

# action normal distribution
def gaussian(mu, sigma):
  return lambda x: 1 / np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2/(2*sigma**2))
fn_norm = gaussian(6.5, args.norm_sigma)
action_space = np.arange(13)
action_prob = fn_norm(action_space); action_prob /= action_prob.sum()

class SecretRealmEnv:
  def __init__(self, env):
    self.env = env
    self.observation_space = (args.obs_dim+1,)  # 1: flash avail
    self.action_space = ()
    self.episodes = self.total_timestep = 0
    self.treasure_miss_cnt = np.zeros((13,), np.int32)
  
  def reset(self):
    self.episodes += 1
    self.n_step = self.total_reward = self.total_score = \
      self.total_hit_wall = self.miss_treasure = 0
    if self.episodes % args.treasure_miss_reset_episode == 0:
      self.treasure_miss_cnt = np.zeros_like(self.treasure_miss_cnt)
    n_treasure = args.n_treasure
    if isinstance(n_treasure, int):
      n_treasure = n_treasure
    elif isinstance(n_treasure, list):  # use ratio selection mode
      block_size = args.total_timesteps // len(n_treasure)
      n = self.total_timestep // block_size
      n_treasure = n_treasure[min(n, len(n_treasure)-1)]
    if n_treasure == 'norm':  # use gaussian distribution
      n_treasure = int(np.random.choice(action_space, 1, p=action_prob)[0])
    self.n_treasure = n_treasure
    self.usr_conf = {
      'diy': {
        'start': 2,  # range in [1, 15]
        'end': 1,  # range in [1, 15], diff with start
        'treasure_random': 1,  # if toggled, choosing treasure positions randomly
        'treasure_num': self.n_treasure,
        'max_step': 2000,
        'talent_type': 1,  # can't change
      }
    }
    obs = self.env.reset(self.usr_conf)
    while obs is None:
      obs = self.env.reset(self.usr_conf)
    obs_feature = observation_process(obs).feature
    obs = obs2dict(obs)
    self.done = False
    self._obs = obs
    self.treasure_reward_coef = np.zeros(13, np.float32)
    if obs['treasure_flags'].sum() != 0:
      self.treasure_reward_coef = np.maximum(
         (self.treasure_miss_cnt+1) /
         (self.treasure_miss_cnt[obs['treasure_flags']]+1).mean(),
         0.5, dtype=np.float32)
    return obs_feature
  
  def step(self, action):
    """
    Args: 
      action: [np.ndarray, shape=(2,)]
        action[0] is direction in 0~7, action[1] is whether using flash
    """
    self.n_step += 1
    self.total_timestep += 1
    if self.done:
      return self.reset(), 0, False, False, {}
    self.action = action
    # time1 = time.time()
    frame_no, obs, score, terminated, truncated, _env_info = self.env.step(self.action)
    # print("[DEBUG] self.env.step time", time.time() - time1)
    total_score = score.score
    score = int(total_score - self.total_score)
    self.total_score = total_score
    obs_feature = observation_process(obs).feature
    self.obs = obs = obs2dict(obs)
    self.done = terminated | truncated
    self.info = info = info2dict(_env_info)
    self.on_buff = self.info['buff_remain_time'] > 0
    self.use_flash = (self._obs['legal_act'][1] - self.obs['legal_act'][1] == 1)
    hit_wall = self._check_hit_wall()

    ### Reward ###
    r = 0
    # 1. punish repeated step around (sum(-weight*max(0, repeat_time-1) * 0.1))
    # ratio = self.total_timestep / args.total_timesteps
    # if ratio < 0.5:
    #   r -= max(obs['memory_map'][25,25] - args.repeat_step_thre, 0)
    #   # assert obs['memory_map'][25,25] > 0  # won't be > 0
    # else:
    #   r -= (args.repeat_punish * np.maximum(
    #     obs[23:28,23:28]-args.repeat_step_thre, 0)).sum()
    # # 2. go to end
    # if terminated:
    #   r += 150
    #   # punish treasures haven't get
    #   r -= (obs['treasure_flags'] * self.treasure_reward_coef).sum() * 50
    #   # punish buff haven't get
    #   r -= obs['buff_flag'] * args.forget_buff_punish
    #   self.treasure_miss_cnt += obs['treasure_flags'].astype(np.int32)
    # dist_reward_coef = args.flash_dist_reward_coef if self.use_flash else args.dist_reward_coef
    # delta_end_distance = self._obs['end_pos']['grid_distance'] - obs['end_pos']['grid_distance']
    # r += delta_end_distance * dist_reward_coef
    # # 3. treasure
    # if not terminated and score == 100:
    #   r += 50 * (self.treasure_reward_coef * (self._obs['treasure_flags'] ^ obs['treasure_flags'])).sum()
    # dist_treasure = np.max(self._obs['treasure_grid_distance'] - self.obs['treasure_grid_distance'])
    # r += dist_treasure * dist_reward_coef
    # 4. hit wall
    if hit_wall:
      r -= args.flash_dist_reward_coef if self.use_flash else args.walk_hit_wall_punish
    # 5. buff
    # if self._obs['buff_flag'] - obs['buff_flag'] == 1:
    #   r += args.get_buff_reward

    self._obs = obs
    self.total_reward += r
    self.total_score += score
    self.total_hit_wall += int(hit_wall)
    self.miss_treasure = sum(obs['treasure_flags'])
    return obs_feature, r, terminated, truncated, info
  
  def _check_hit_wall(self):
    if not self._check_has_wall_around(): return False
    _abs_pos, abs_pos = [x['norm_pos'] * 64000 for x in [self._obs, self.obs]]
    dist = np.linalg.norm(_abs_pos - abs_pos)
    if (not self.on_buff and dist < const.delta_dist_walk or
        self.on_buff and dist < const.delta_dist_buff or
        self.use_flash and dist < const.delta_dist_flash):
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
  treasure_flags, treasure_grid_distance = np.zeros(13, bool), np.zeros(13, np.float32)
  for i, treasure in enumerate(obs['treasure_pos'][2:]):
    treasure_flags[i] = treasure['direction'] != 0
    treasure_grid_distance[i] = treasure['grid_distance']
  buff_flag = obs['buff_pos']['direction'] != 0
  obs.update({
     'treasure_flags': treasure_flags,
     'treasure_grid_distance': treasure_grid_distance,
     'buff_flag': buff_flag
  })
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
  obs = obs2dict(raw_obs)
  feature = np.hstack([
    np.stack([  # Image: (4, 51, 51)
      obs['obstacle_map'],
      obs['memory_map'],
      obs['treasure_map'],
      obs['end_map'],
    ], axis=0).reshape(-1),
    *obs['norm_pos'],
    *obs['treasure_flags'],
    *obs['treasure_grid_distance'],
    obs['buff_flag'],
    obs['buff_pos']['grid_distance'],
    obs['end_pos']['grid_distance'],
    obs['legal_act'][1],  # flash available
  ]).astype(np.float32)
  assert feature.shape[0] == args.obs_dim + 1, f"ERROR: {feature.shape[0]=} != {args.obs_dim+1}"
  return ObsData(feature=feature)

@attached
def action_process(act_data):
    return act_data.act

@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]
