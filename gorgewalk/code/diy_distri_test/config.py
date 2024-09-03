import numpy as np

# Configuration
Args = {
    "version": "1.3beta20_8e6",  # model version
    "seed": 1,  # seed of the experiment
    "torch_deterministic": True,  # toggles pytorch's cudnn.deterministic 卷积算法是否相同
    "cuda": False,  # cuda
    "capture_video": False,  # captures video

    # Algorithm specific arguments
    "total_timesteps": int(8e6),  # total timesteps of the experiments
    "learning_rate":  2.5e-5,  # the learning rate of the optimizer 这个可以在稳定之后下降，比如除以十
    # "group_lr": [2.5e-4, 2.5e-5],  # adjust learning rate by the proportion 按照步数比例调整学习率(在不启动退火的前提下)
    "group_lr": [2.5e-4, 2.5e-5, 2.5e-5],  # adjust learning rate by the proportion 按照步数比例调整学习率(在不启动退火的前提下)
    "num_envs": 1,  # the number of parallel environments
    "num_steps": 512, # the number of steps to run in each environment per policy rollout 这个最好大于一个episode的长度，设置成512或者1024
    "anneal_lr": False, # whether to anneal the learning rate or not 是否退火
    "gamma": 0.999, # the discount factor gamma 这个要再高一些 0.999左右，也可以是1
    "gae_lambda": 0.95,  # the lambda for the general advantage estimation
    "num_minibatches": 16, # the number of mini-batches
    "update_epochs": 4, # the K epochs to update the policy
    "norm_adv": False, # advantages normalization 可能不需要
    "clip_coef": 0.2, # the surrogate clipping coefficient
    "clip_vloss": True, # whether or not to use a clipped loss for the value function, as per the paper
    "ent_coef": 1e-4, # entropy coefficient 一般设置成1e-4，在训练最优模型的时候设置成0
    #"group_ent_coef": [1e-2, 1e-4], # adjust entropy coefficient by proportion
    "group_ent_coef": [1e-2, 1e-4, 1e-4], # adjust entropy coefficient by proportion
    "vf_coef": 0.5, # value function coefficient
    "max_grad_norm": 0.5, # the maximum norm for the gradient clipping
    "target_kl": None, # the target KL divergence threshold

    # to be filled in runtime
    "batch_size": 0, # the batch size (computed in runtime) 
    "minibatch_size": 0, # the mini-batch size (computed in runtime)
    "num_iterations": 0, # the number of iterations (computed in runtime)

    # Network
    "obs_dim": 249,
    # "hid_units": [256, 256],
    "hid_units": [128, 128],
    # "hid_units": [128, 64],
    # "hid_units": [64, 32],
    # Environment
    #"n_treasure": [10, 'norm', 10, 'norm'],
    # "n_treasure": ['norm', 10, 'norm'],
    "n_treasure": ['norm'],
    # "n_treasure": [*range(11), 'norm'],
    "norm_sigma": 2,
    # "n_treasure": 5,
    # Reward
    "dist_reward_coef": 0.1,
    "repeat_punish": np.array([
        [0, 0, 0, 0, 0],
        [0, 0.5, 0.8, 0.5, 0],
        [0, 0.8, 1.0, 0.8, 0],
        [0, 0.5, 0.8, 0.5, 0],
        [0, 0, 0, 0, 0],
    ], np.float32),
    "treasure_miss_reset_episode": 100,
    "repeat_step_thre": 0,
}

class Config:

    # dimensionality of the sample
    # 样本维度
    SAMPLE_DIM = 1

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