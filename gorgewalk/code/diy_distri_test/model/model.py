import torch
import numpy as np
from torch import nn
from diy.config import Args
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Model(nn.Module):
    def __init__(self, obs_dim=Args['obs_dim']):
        super().__init__()
        self.obs_dim = obs_dim
        hid_units = Args['hid_units']
        assert len(hid_units) == 2
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hid_units[0])),
            nn.Tanh(),
            layer_init(nn.Linear(hid_units[0], hid_units[1])),
            nn.Tanh(),
            layer_init(nn.Linear(hid_units[1], 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hid_units[0])),
            nn.Tanh(),
            layer_init(nn.Linear(hid_units[0], hid_units[1])),
            nn.Tanh(),
            layer_init(nn.Linear(hid_units[1], 4), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
