import torch
import numpy as np
from torch import nn
from diy.config import args
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
  if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
      nn.init.constant_(layer.bias, bias_const)
  return layer

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.cnn = nn.Sequential(
      nn.Conv2d(4, 32, kernel_size=7, stride=2),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=5, stride=2),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(4096, 512),
      nn.ReLU()
    )
    self.fc = nn.Sequential(
      nn.Linear(512+args.observation_vec_shape[0], 512),
      nn.ReLU()
    )
    self.apply(layer_init)
    self.actor = layer_init(nn.Linear(512, 16), std=0.01)
    self.critic = layer_init(nn.Linear(512, 1), std=1)
  
  def get_latent(self, x):
    obs_size = np.prod(args.observation_img_shape)
    B = x.shape[0]
    img, vec = x[:, :obs_size], x[:, obs_size:]
    img = img.view(B, *args.observation_img_shape)
    x = torch.cat([self.cnn(img), vec], -1)
    return self.fc(x)

  def get_value(self, x):
    x = x[:, :-1]
    return self.critic(self.get_latent(x))

  def get_action_and_value(self, x, action=None):
    B = x.shape[0]
    x, flash = x[:, :-1], x[:, -1:]
    mask = torch.cat([torch.ones(B, 8, device=flash.device, dtype=bool), flash.expand(B, 8).bool()], -1)
    z = self.get_latent(x)
    logits = self.actor(z)
    logits = torch.where(~mask, logits, -1e8)
    probs = Categorical(logits=logits)
    if action is None:
      action = probs.sample()
    return action, probs.log_prob(action), probs.entropy(), self.critic(z)

if __name__ == '__main__':
  B = 32
  img = torch.randn(B, *args.observation_img_shape)
  vec = torch.randn(B, *args.observation_vec_shape)
  mask = torch.tensor([0]*8+[1]*8, dtype=torch.bool)
  mask = mask.expand((B, 16))
  print(mask.shape)
  print(img.shape, vec.shape)
  x = torch.cat([img.view(B, -1), vec.view(B, -1), torch.ones(B, 1)], -1)
  print(x.shape)
  model = Model()
  value = model.get_value(x)
  print(value.shape)
  action, logprob, ent, value = model.get_action_and_value(x)
  print(action.shape, logprob.shape, ent.shape, value.shape)
