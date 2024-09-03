from kaiwu_agent.utils.common_func import create_cls, attached
import numpy as np
from diy.utils import ProcessPrinter

# SampleData的格式要和NumpyData2SampleData中一样（这里以SARST为例）
SampleData = create_cls("SampleData", obs=None, act=None, rew=None, _obs=None, terminal=None)
# PPO
# SampleData = create_cls("SampleData", obs=None, logprob=None, act=None, adv=None, ret=None, val=None)
ObsData = create_cls("ObsData", feature=None)
ActData = create_cls("ActData", act=None)

@attached
def observation_process(raw_obs):
    return ObsData(feature=raw_obs)

@attached
def action_process(act_data):
    return act_data.act

@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]

@attached  # 这里就是要将dict数据结构张成一位向量保存，猜测buffer是最简单的矩阵存储形式，所以需要如此转换
def SampleData2NumpyData(data):
    return np.hstack([  # 用 np.concatenate 应该一样
        np.array(data.obs, dtype=np.float32),  # 250
        np.array(data.act, dtype=np.float32),  # 1
        np.array(data.rew, dtype=np.float32),  # 1
        np.array(data._obs, dtype=np.float32),  # 250
        np.array(data.terminal, dtype=np.float32),  # 1
    ])

@attached  # 这里的转换要和SampleData2NumpyData中输出的格式保持一致
def NumpyData2SampleData(data):
    process_printer = ProcessPrinter()
    process_printer(f"NumpyData2SampleData: {len(data)=}, {data.shape=}")
    return SampleData(
        obs=data[:250],
        act=data[250:251],
        rew=data[251:252],
        _obs=data[252:502],
        terminal=data[502:503]
    )