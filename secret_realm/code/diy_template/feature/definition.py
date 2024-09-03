from kaiwu_agent.utils.common_func import create_cls, attached

# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None
ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_act=None,
)


ActData = create_cls(
    "ActData",
    move_dir=None,
    use_talent=None,
)


SampleData = create_cls(
    "SampleData",
    obs=None,
    _obs=None,
    obs_legal=None,
    _obs_legal=None,
    act=None,
    rew=None,
    ret=None,
    done=None,
)


def reward_shaping(frame_no, score, terminated, truncated, obs, _obs, env_info, _env_info):
    pass


@attached
def observation_process(raw_obs, env_info=None):
    pass


@attached
def action_process(act_data):
    pass


@attached
def sample_process(list_game_data):
    pass


# SampleData <----> NumpyData
@attached
def SampleData2NumpyData(g_data):
    pass


@attached
def NumpyData2SampleData(s_data):
    pass
