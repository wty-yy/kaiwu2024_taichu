from kaiwu_agent.utils.common_func import create_cls, attached
import random
import numpy as np
from diy.config import Args
SampleData = create_cls("SampleData", state=None, action=None, reward=None)
ObsData = create_cls("ObsData", feature=None)
ActData = create_cls("ActData", act=None)

action2delta = np.array([(0, 1), (0, -1), (-1, 0), (1, 0)], np.int32)

# action normal distribution
def gaussian(mu, sigma):
  return lambda x: 1 / np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2/(2*sigma**2))
fn_norm = gaussian(5, Args['norm_sigma'])
action_space = np.arange(11)
action_prob = fn_norm(action_space); action_prob /= action_prob.sum()

class GorgeWalk():
    def __init__(self, env, random_env=True, max_step=2000):
        self.env, self.random_env, self.max_step = env, random_env, max_step
        self.observation_space = (Args['obs_dim'],)
        self.action_space = ()  # scale, not (4,) !!!
        self.episodes, self._total_timestep = 0, 0
        self.dist_reward_coef = Args['dist_reward_coef']
        self.treasure_miss_cnt = np.zeros((10,), np.int32)

    def reset(self):
        self.episodes += 1
        if self.episodes % Args['treasure_miss_reset_episode'] == 0:
            self.treasure_miss_cnt = np.zeros_like(self.treasure_miss_cnt)
        n_treasure = Args['n_treasure']
        if isinstance(n_treasure, int):
            n_treasure = n_treasure
        elif isinstance(n_treasure, list):  # use ratio selection mode
            block_size = Args['total_timesteps'] // len(n_treasure)
            n = self._total_timestep // block_size
            n_treasure = n_treasure[min(n, len(n_treasure)-1)]
        if n_treasure == 'norm':  # use gaussian distribution
            n_treasure = int(np.random.choice(action_space, 1, p=action_prob)[0])
            # n_treasure = int(np.clip(np.random.randn(1)[0] * 2, -5, 5).astype(np.int32) + 5)  # old norm distri, bigger sigma
        self.n_treasure = n_treasure
        assert isinstance(n_treasure, int), f"Can't parse {Args['n_treasure']=}"
        self.usr_conf = {
            "diy": {
                "start": [29, 9],
                "end": [11, 55],
                "treasure_num": n_treasure,
                "treasure_random": 1,
                "max_step": self.max_step,
            }
        }
        self._step = 0
        obs = self.env.reset(usr_conf=self.usr_conf)
        while obs is None:
            obs = self.env.reset(usr_conf=self.usr_conf)
        self.total_reward, self.total_score, self.total_length, self._done = 0, 0, 0, False
        self.hit_wall = 0
        self._last_pos = obs[0]
        self._last_treasure_flag = np.array(obs[240:250], bool)
        if self._last_treasure_flag.sum() == 0:
            self._last_treasure_flag ^= True  # anything is ok, since it will not be used
        self.treasure_reward_coef = np.maximum((self.treasure_miss_cnt+1) / (self.treasure_miss_cnt[self._last_treasure_flag]+1).mean(), 0.5)
        return observation_process(obs).feature, {}

    def step(self, action: int):
        self._step += 1
        self._total_timestep += 1
        if self._done:
            return self.reset()[0], 0, False, False, {}
        frame_no, obs, score, terminated, truncated, env_info = self.env.step(action)
        obs_data = observation_process(obs).feature
        self._done = terminated or truncated
        ### Reward ###
        r = 0
        # 1. punish repeated step around (sum(-weight*max(0, repeat_time-1) * 0.1))
        ratio = self._total_timestep / Args['total_timesteps']
        # repeat_thre, thre_group = 1.0, Args['repeat_thre_group']
        # if len(thre_group):
        #     repeat_thre = thre_group[int(ratio*len(thre_group))]
        # r -= max(obs_data[214+12] - 0.1 * Args['repeat_step_thre'], 0) * repeat_thre
        if ratio < 0.5:
            r -= max(obs_data[214+12] - 0.1 * Args['repeat_step_thre'], 0)
            assert obs_data[214+12] > 0
        else:
            r -= (Args['repeat_punish'] * np.maximum(
                obs_data[214:239]-0.1*Args['repeat_step_thre'], 0).reshape(5, 5)).sum()
        # 2. go to end
        if terminated:
            r += 150
            # punish treasures haven't get
            r -= (obs_data[239:249] * self.treasure_reward_coef).sum() * 50
            self.treasure_miss_cnt += obs_data[239:249].astype(np.int32)
        ant_dist_end = obs_data[128]
        r += ant_dist_end * self.dist_reward_coef
        # 3. treasure
        if not terminated and score == 100:
            r += 50 * (self.treasure_reward_coef * (self._last_treasure_flag - obs_data[239:249])).sum()
        ant_dist_treasure = np.max(obs_data[129:139])
        if ant_dist_treasure != -1:
            r += ant_dist_treasure * self.dist_reward_coef
        # 4. don't move (hit wall)
        if obs[0] == self._last_pos:
            r -= 1
            self.hit_wall += 1

        self._last_pos = obs[0]
        self._last_treasure_flag = obs_data[239:249]
        self.total_reward += r
        self.total_score += score
        self.total_length += 1
        self.exist_treasure = np.sum(obs_data[239:249])
        return obs_data, r, terminated, truncated, {}

@attached
def observation_process(raw_obs):
    """
    [0]: 智能体当前位置状态,state = x * 64 + z
    [1: 65]: 智能体当前位置横坐标的 one-hot 编码
    [65: 129]: 智能体当前位置纵坐标的 one-hot 编码
    [129: 130]: 智能体当前位置相对于终点的离散化距离,0-6, 数字越大表示越远
    [130: 140]: 智能体当前位置相对于宝箱的离散化距离,0-6, 数字越大表示越远
    [140: 165]: 智能体局部视野中障碍物的信息 (一维化),1 表示障碍物, 0 表示可通行
    [165: 190]: 智能体局部视野中宝箱的信息 (一维化),1 表示宝箱, 0 表示没有宝箱
    [190: 215]: 智能体局部视野中终点的信息 (一维化),1 表示终点, 0 表示非终点
    [215: 240]: 智能体局部视野中的记忆信息 (一维化),取值范围[0,1], 一个格子每走过一次, 记忆值+0.1
    [240: 250]: 宝箱的状态,1 表示可以被收集, 0 表示不可被收集(未生成或者已经收集过), 长度为 10
    """
    obs = np.array(raw_obs, np.float32)
    # obs[129:140] /= 6  # 0~6 -> 1~0
    flag = obs[240:250].astype(bool)
    # end distance 0~6 -> 1~0
    obs[129] = 1 - obs[129] / 6
    # treasure distance 0~6 and 999 -> 1~0 and -1 (999 means no treasure)
    treasure_dist = obs[130:140]
    treasure_dist[~flag] = -1
    treasure_dist[flag] = 1 - treasure_dist[flag] / 6
    # print("DEBUG:", obs[129:140], obs[240:250])
    obs = obs[1:]
    return ObsData(feature=obs)  # shape=(249,)
    # return ObsData(feature=obs[140:165])

@attached
def action_process(act_data):
    return act_data.act

@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]
