## 20240716
### v1.0
神经网络中间层神经元数目均为 `32`，奖励函数设计如下
```python
self.dist_reward_alpha = 0.1
r = 0
# 1. step punish
r -= 0.2
# 2. go to end
if terminated:
    r += 150
ant_dist_end = obs_data[128]
r += ant_dist_end * self.dist_reward_alpha
# 3. treasure
if not terminated and score == 100:
    r += 100
ant_dist_treasure = np.max(obs_data[129:139])
if ant_dist_treasure != -1:
    r += ant_dist_treasure * self.dist_reward_alpha
# 4. don't move (hit wall)
if obs[0] == self._last_pos:
    r -= 1
    self.hit_wall += 1
```
### v1.1
奖励函数进行两个修改：
```python
# 1. step punish 只对重复走的位置进行惩罚，惩罚为重复次数*0.1 (UPDATE)
rpos = action2delta[action] + np.array([2, 2], np.int32)  # position relative to previous step 5x5 center
repeat_time = self._last_memory[rpos[0], rpos[1]]
if repeat_time > 0:
    r -= repeat_time
# 2. go to end
if terminated:
    r += 150
    # punish treasures haven't get (UPDATE) 对未达到的宝藏进行惩罚
    r -= np.sum(obs_data[239:249]) * 50
# 3. treasure
if not terminated and score == 100:
    r += 50  # UPDATE: 100 -> 50
```
### v1.2
1. 神经网络中间层结构改为 `249-32-32-* ->  249-128-64-*`
2. 使用动态调节的环境宝箱数初始化 `[10, norm, 10, norm]`，将全部的 `total_timestep` 等比例进行划分，每次按照划分对应的初始化进行宝箱数目选择。
## 20240717
### v1.3
更新奖励函数：
1. 重复步数惩罚改为范围加权形式（不对首次到达的位置惩罚）
2. 对宝箱按照其未捡到的次数进行正比例惩罚
```python
# config.py
Args = {
    ...,
    "repeat_punish": np.array([
        [0, 0, 0, 0, 0],
        [0, 0.5, 0.8, 0.5, 0],
        [0, 0.8, 1.0, 0.8, 0],
        [0, 0.5, 0.8, 0.5, 0],
        [0, 0, 0, 0, 0],
    ], np.float32),
    "treasure_miss_reset_episode": 10,
    ...
}

# agent.py
# in step()
# 1. punish repeated step around (sum(-weight*max(0, repeat_time-1) * 0.1))
r -= (Args['repeat_punish'] * np.maximum(obs_data[214:239]-0.1, 0).reshape(5, 5)).sum()
# 2. go to end
if terminated:
    r += 150
    # punish treasures haven't get
    r -= (obs_data[239:249] * self.treasure_reward_coef).sum() * 50  # with coef
    self.treasure_miss_cnt += obs_data[239:249].astype(np.int32)
ant_dist_end = obs_data[128]
r += ant_dist_end * self.dist_reward_alpha
# 3. treasure
if not terminated and score == 100:
    r += 50 * (self.treasure_reward_coef * (self._last_treasure_flag - obs_data[239:249])).sum()  # with coef

# in reset()
self.treasure_reward_coef = np.maximum((self.treasure_miss_cnt+1) / (self.treasure_miss_cnt[self._last_treasure_flag]+1).mean(), 0.5)
```

## 20240718
### v1.3beta1
单独测试5个随机宝箱的水平。
1. 使用更小的网络结构 `249-128-64-* -> 249-32-32-*`
2. 使用更大的学习率 `2.5e-4 -> 5e-4`
3. `total_timesteps=2e6`
### v1.3_beta2
1. 网络结构 `249-64-32-*`
2. 学习率 `2.5e-4`
分层奖励：前1/2的训练过程中，只对当前重复走过位置进行惩罚，后1/2的再对周围3x3的重复位置进行惩罚。
### v1.3_beta3
调整奖励：
1. 周围3x3重复走过位置，只有当重复次数超过2次时，才产生惩罚。
2. 上调 `"treasure_miss_reset_episode": 10 -> 100`
### v1.3_beta4
调回网络结构为 `249-128-64-*`
### v1.3_beta5
测试`1e7`步训练结果

## 20240719
### v1.3_beta6
测试增大网络结构 `249-128-128-*`，测试 `5e6` 步训练结果，宝箱数按照 `[10, norm, 10, norm]` 进行训练。
### v1.3_beta7
继续增大网络结构 `249-256-256-*`
### v1.3_beta8
网络结构调回 `249-128-128-*`，与beta6完全相同的参数（本来想用beta9增大方差的方法），但在验证效果上略差于beta6，说明不稳定。

## 20240720
### v1.3_beta9
正态分布函数重新进行修改，放弃获取随机数后取int的方法（边缘值0,10的访问次数太少），而是使用对应概率分布直接选择的方法，高斯分布为 `mu=5,sigma=3`：
```python
def gaussian(mu, sigma):
  return lambda x: 1 / np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2/(2*sigma**2))
fn_norm = gaussian(5, Args['norm_sigma'])
action_space = np.arange(11)
action_prob = fn_norm(action_space); action_prob /= action_prob.sum()
# 选择
n_treasure = int(np.random.choice(action_space, 1, p=action_prob)[0])
```
> beta9除了在5个宝箱的上面效果比beta6差，其他都可以
### v1.3_beta10
尝试新的宝箱分布 `[*range(11), 'norm']`

## 20240721
### v1.3_beta11
1. beta10的新分层训练方法效果很差，还是采用 `[10, 'norm', 10, 'norm']` 的宝箱分层
2. 减小norm的标准差为2
3. 增大PPO中的折扣系数 `gamma=0.99 -> 0.999`
4. 按照 `[2.5e-4, 2.5e-5]` 比例调整学习率
5. 关闭 `norm_adv`
6. 按照 `[1e-2, 1e-4]` 比例调整ent_coef
7. 增大 `num_steps=128 -> 512`
### v1.3_beta12
增大`num_minibatches=4 -> 16`，即减小 `batchsize=128 -> 32`
### v1.3_beta13
加入新超参数`repeat_step_thre`，开始对重复步数进行惩罚的最小阈值，将其从 `2 -> 0`，将训练步数增加到2e7
## 20240722
### v1.3_beta14
将训练步数重新减小到 `5e6`
### v1.3_beta15
将训练步数重新增加到 `1e7`
### v1.3_beta16
直接使用小学习率2.5e-5，小正则项系数1e-4，训练步数减小到5e6
## 20240723
### v1.3_beta17
尝试"n_treasure": [10, 'norm']
### v1.3_beta18
尝试"n_treasure": ['norm', 10, 'norm'], 对应学习率为[2.5e-4, 2.5e-5, 2.5e-5], ent_coef为[1e-2, 1e-4, 1e-4]
### v1.3_beta19
尝试`n_treasure=['norm']`
## 20240724
### v1.3_beta20
增大`total_timesteps=int(8e6)`
### v1.3_beta21
减小`total_timesteps=int(6e6)`
### v1.3_beta22
1. 增大低学习率比例, `"group_lr": [2.5e-4, 2.5e-5, 2.5e-5] -> [2.5e-4, 2.5e-5, 2.5e-5, 2.5e-5]`
2. 增大ent coef低比例 `"group_ent_coef": [1e-2, 1e-4, 1e-4] -> [1e-2, 1e-4, 1e-4, 1e-4]`
## 20240725
### v1.3_beta20
继续测试beta20版本
## 20240726
### v1.3_beta23
重新使用分段，关闭距离奖励
1. `"group_lr": [2.5e-4, 2.5e-5, 2.5e-5] -> [1e-4, 5e-5, 2.5e-5, 2.5e-5]`
2. `"group_ent_coef": [1e-2, 1e-4, 1e-4] -> [1e-2, 1e-3, 1e-4, 1e-5]`
3. `"n_treasure": ['norm'] -> [10, 'norm', 10, 'norm']`
4. `"dist_reward_coef": 0.1 -> 0`
### v1.3_beta24
1. 打开 `norm_adv`
2. 减小 `num_minibatches: 16 -> 8`，即增大 `batchsize: 32 -> 64`，这样会提高训练速度 4:20 -> 3:50
### v1.3_beta25
进一步减小正则系数，减小学习率
1. "n_treasure": [10, 'norm', 10, 'norm'] -> [10, 'norm', 10, 'norm', 'norm']
2. "group_ent_coef": [1e-2, 1e-3, 1e-4, 1e-5] -> [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
3. "group_lr": [1e-4, 5e-5, 2.5e-5, 2.5e-5] -> [1e-4, 5e-5, 2.5e-5, 1e-5, 1e-6]
4. "total_timesteps": int(8e6) -> int(1e7)
## 20240728
### v1.3_beta26
1. 增大后期步数惩罚
2. 关闭norm_adv