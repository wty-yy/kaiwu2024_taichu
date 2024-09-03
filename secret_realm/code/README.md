# 重返秘境
## Debug信息
距离相关信息: 环境中离散化网格大小为 `128x128`, 每个网格被细分为 `500` 码距离, 因此总网格大小为 `64000x64000`
1. 每步移动的距离, 没有移速加成的情况下 660~711, 有移速buff的情况下 900~1100, 闪现移动距离 7900~8100, 通过这些可以判断是否撞墙, 代码中处于 `diy.feature.constants.py` 文件下.
2. 模型的obs输出的map是有上下翻转的.
3. buff的使用时长是50步, 捡到之后不再会刷新新的buff了.
4. 闪现的cd为600步, 一个episode最大基本为550步, 因此基本不会使用到两次闪现的.
5. `env.step` 返回的 `score` 是一个类, 而不是一个 `int`, 通过 `score.score` 获得到的得分是当前总宝箱的得分.
### obs信息解包
分别对 `obs` 和 `info` 进行解包, 包中包含的属性可以在官网 [数据协议](https://doc.aiarena.tencent.com/competition/back_to_the_realm/1.0.0/guidebook/protocol/) 中查到, 代码中见 `diy.feature.definition.py` 中的 `obs2dict, info2dict` 函数.
### 网络结构设计
代码请见 `diy.algorithm.model.py` 中的 `Model` 类, 对于 `actor, critic` 我们分别使用了两个 backbone (可以测试共享 backbone):
- 卷积仅使用的是最朴素的3层CNN+ReLU, 图像输入维度为 `(4, 51, 51)`, 就是 `obs` 中返回的图像堆叠而成.
- 状态输入, 维度为 `(31,)`: Agent归一化后的位置 `(2,)`, 宝箱的存在标志 `(13,)`, 宝箱grid_distance `(13,)`, buff存在标志 `(1,)`, buff的grid_distance `(1,)`, 终点的grid_distance `(1,)`.
动作空间为 `(16,)`, 前8维度为正常移动, 后8维度为使用闪现的移动, 移动都是8个方向.
## Distribution PPO
在一院这边从8月2日开始学习源码 (源码学习请见 [代码逻辑](../../assets/code_logic.md)), 直到8月11日完成分布式PPO代码, 主要注意如下内容:
1. replay buffer的sample模式调试, 最终还是选择 `Uniform` 模型
2. buffer中的每条信息是一个连续的轨迹信息 (s, a, r, done, last_s, last_done), 这6个信息, 轨迹长度设置为 `num_steps`, 在learner中通过 GAE 优势函数, 用策略网络计算 `logprob`
3. `agent` 中不能存储环境, 也就是不能把 `rollout` 函数放到 `agent` 类中, 因为所有的 `aisrv` 是共享同一个 `agent`, 我们只能通过 `predict` 函数去做动作预测 (这也是一个通讯过程), 所以 `agent` 和环境交互的过程全部要在 `workflow` 中完成.
4. 模型同步细节, 由于PPO要求模型的高同步性, 因此我将模型的自动保存次数设置为 `dump_model_freq=2`, 模型权重文件同步时间设置为 `model_file_sync_per_minutes=1` (这里最小单位就是分钟, 最小就是1分钟)
5. 模型保存: 具体保存细节参见 [代码逻辑 - save_model逻辑](../../assets/code_logic.md#save_model逻辑), 简单就是, 由于 `learner` 和 `actor` 之间的模型有延迟, 因此中间有个模型池专门用来同步 `learner` 中的最新模型, 而模型池的模型同步只会从周期性自动保存的模型中进行获取 (也就是 `dump_model_freq` 设置的周期), 注意这个保存的模型不会保存在本地. 而我们的手动调用 `agent.save_model()` 函数, 模型才会保存在本地, 而这个保存有个最大限制次数 `200`, 我们可以通过设置 `user_save_mode_max_count = 0` 就是无限制保存次数, 为了避免空间爆炸, 在保存模型后, 执行以下函数删掉之前旧的模型节省空间 (默认保存最近的10个节点, 使用时只需将 `logger` 传入即可打印日志):
```python
from pathlib import Path
from kaiwudrl.common.config.config_control import CONFIG

def clean_ckpt_memory(min_num_model=10, logger=None):
  """
  Remove old checkpoints generate by autosave (save frequency=CONFIG.dump_model_freq),
  you can find autosave code at
  `kaiwudrl.common.algorithms.standard_model_wrapper_pytorch.StandardModelWrapperPytorch.after_train()`

  Usage: Call this function in `agent.learn(...)` or `train_workflow.workflow`, 
  recommend in `agent.learn(...)` since it is a single process,
  add small delay as you like~~~
  """
  path_tmp_dir = Path(f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/")
  files = sorted(list(path_tmp_dir.glob('model.ckpt-*')), key=lambda x: int(str(x).rsplit('.', 1)[0].rsplit('-', 1)[1]))
  if len(files) > min_num_model:
    info = f"Remove old ckpts ({path_tmp_dir}): "
    for p in files[:-min_num_model]:  # just keep latest checkpoint
      p.unlink()
      info += f"{p.stem} "
    if logger is not None:
      logger.info(info)
```
### 训练配置细节
PPO的重要参数包含以下几个, 我这以8进程启动为例:
```toml
# 关闭死循环训练, 而是基于有多少新加入的样本数目开始训练
learner_train_by_while_true = false
# 默认就是 off-policy, 当前也不支持 on-policy, 
# 这个主要影响buffer启动训练的次数
algorithm_on_policy_or_off_policy = 'off-policy'
# buffer的大小, 这个和客户端启动的进程数相关, 
# 如果是采样模型是Uniform最好是进程数*2~4, 
# 如果是Fifo那么写进程数就行, 因为每次就取的是最后一个
replay_buffer_capacity = 24
# 当buffer中存在replay_buffer_cAapacity/preload_ratio个样本时,
# 开始训练, 我配置塞满了就开始训
preload_ratio = 1  
# learner中每次向agent.learn传入的样本数目, 
# Uniform模式建议就是进程数, Fifo模型只能写1, 因为他只会取最后一个
train_batch_size = 8
# 样本消耗/生成采样比
# 和offline的训练次数有关, 当上次训练后新加进来的样本数目超过
# train_batch_size / production_consume_ratio时候再开始一次训练
production_consume_ratio = 1
# 采样策略选择
# Fifo只会选取最后一个样本, 导致整个batch里面全是一样的, 只有在train_batch_size=1有用
# 推荐Uniform均匀采样
reverb_sampler = "reverb.selectors.Uniform"
# 训练间隔多少步输出model文件, 这个就是可以同步到模型池的自动保存频率
dump_model_freq = 2
```

## 2024.8.12.
### v0.4
固定起点2终点1, 宝箱数目为[13, 'norm', 13, 'norm'], 训练1e7步, 同样也遇到了训练一段时间后, value直接崩溃的问题.
### v0.4.1
讨论后考虑有如下这些改进方向, 逐步进行尝试, 看能否解决问题:

- [x] return过大, 尝试直接 /10 进行缩放
- [x] logprob 应该是由actor给出传到buffer里面, 而非通过learner重新计算
- [ ] value 是否应该也是由actor给出
- [x] 增大`buffer size`到`进程数*10`, 增大`batchsize`到进程数*3, 增大消耗/生成比`production_consume_ratio=3`, 预加载比例 `preload_ratio=2`
- [x] 日志中输出clip后的`grad_norm`查看是否会顶满`max_grad_norm`
- [x] 减小学习率 `2.5e-4 -> 5e-5`

## 2024.8.13.
### v0.4.2
1. 修复闪现撞墙的错误奖励
2. env中计算ratio时忘记除以num_envs
3. 减小训练步长 `1e7->5e6`
4. 修复total_score忘记加步数得分的问题
5. 日志中加入buffer相关参数 `miss_buffer`
6. `ent_coef: 1e-2 -> 1e-3`
7. 修复奖励中delta treasure错误计算了缺失的宝箱
8. `repeat_step_thre = 0.2 -> 0.4`
9. `n_treasure = [13,'norm',13,'norm'] -> ['norm', 13, 'norm']`
## 2024.8.14.
### v0.4.3
1. 将 `num_minibatches` 替换为 `minibatch_size`, 直接指定minibatch大小
2. `update_epoches: 2 -> 1`
3. `norm_adv: True -> False`
4. `norm_sigma: 2.0 -> 3.0`, 修正 `action_space: np.arange(13) -> np.arange(14)`
5. 修复env中 `ratio` 计算错误的问题, 能够按比例调整宝箱数目, 修复reward里面周围步数惩罚的错误.
## 2024.8.15.
### v0.4.4
1. 每步都加入 `0.2` 的惩罚
2. `num_envs: 10 -> 12`
3. `num_steps: 128 -> 512`

### v0.4.4.1
训练到47293步时, 出现环境范围为None的报错, 导致训练停止???
```python
202024-08-15 13:54:04.110 | ERROR    | kaiwudrl.server.aisrv.kaiwu_rl_helper_standard:workflow:461 - aisrv kaiwu_rl_helper workflow() Exception 'NoneType' object has no attribute 'score', traceback.print_exc() is Traceback (most recent call last):
  File "/data/projects/back_to_the_realm/kaiwudrl/server/aisrv/kaiwu_rl_helper_standard.py", line 437, in workflow
    AlgoConf[CONFIG.algo].train_workflow([env], self.current_models, self.logger, self.monitor_proxy)
  File "/data/projects/back_to_the_realm/diy/train_workflow.py", line 45, in workflow
    next_obs, reward, terminations, truncations, infos = env.step(action)
                                                         ^^^^^^^^^^^^^^^^
  File "/data/projects/back_to_the_realm/diy/feature/definition.py", line 87, in step
    total_treasure_score = score.score
                           ^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'score'

# score 来源于: frame_no, obs, score, terminated, truncated, _env_info = self.env.step(self.action)
```
加入对score的异常处理, 如果为None则反复进行 `env.step`. 出现该报错的原因可能是 `num_steps` 调到了 `512` ? 重新减小回 `128`
1. `num_steps: 512 -> 128`
2. `num_envs: 12 -> 10`
训练到12h时候再次出现`score=None`的情况 (确认应该和硬盘空间不足有关), 为了能够今天跑完跑榜任务, 特训一个10个宝箱的

## 2024.8.16.
无法开启到客户端指定的进程数目, 原因也可能和硬盘空间不足有关, 尽可能多开一点, 16进程一般只能开到12进程
1. `num_envs: 10 -> 12`
2. `each_step_punish: 0.2 -> 0.1`, 只在后半程训练 `ratio > 0.5` 中加入步数惩罚

## 2024.8.17.
### v1.0
对v0.4.3的4548步模型接着训练:
1. 采用每步 `0.1` 的惩罚
2. `num_envs: 12 -> 10, num_steps: 128 -> 256`
3. 加入随机起点, 在前50%的episode中智能体初始化在随机的起点上, 后50%只能出现在2号节点上.
4. `learning_rate: 5e-5 -> 2.5e-5, ent_coef: 1e-3 -> 1e-4`

## 2024.8.19.
### v1.1
完成v1.0训练, 提升挺大, 部分情况下会出现漏宝箱的问题
1. 关闭动态宝箱奖励, 增大宝箱奖励与惩罚 `50 -> 100`
2. 增大步数惩罚, `0.1 -> 0.2`
3. `n_treasure: norm -> uniform`
4. 关闭随机起点 `random_start_position_ratio = 0.5 -> 0.0`
5. 降低学习率 `2.5e-5 -> 1e-5`
6. 增大 `update_epochs: 1 -> 2`, 
7. 减小 `ent_coef: 1e-4 -> 1e-5`

没内存了😢, 不能增大 `replay_buffer_capacity: 100 -> 200, train_batch_size: 30 -> 60`
