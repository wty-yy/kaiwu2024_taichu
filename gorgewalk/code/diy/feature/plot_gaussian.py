import numpy as np
import matplotlib.pyplot as plt

config = {  # matplotlib绘图配置
  "figure.figsize": (6, 6),  # 图像大小
  "font.size": 16, # 字号大小
  'axes.unicode_minus': False # 显示负号
}
plt.rcParams.update(config)

def gaussian(mu, sigma):
  return lambda x: 1 / np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2/(2*sigma**2))
action_space = np.arange(11)

def show(sigma):
  fn_norm = gaussian(5, sigma)
  action_prob = fn_norm(action_space); action_prob /= action_prob.sum()
  plt.plot(action_space, action_prob, label=f"$\sigma={sigma}$")

show(sigma=1)
show(sigma=2)
show(sigma=3)
plt.legend()
plt.show()