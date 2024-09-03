import numpy as np
import matplotlib.pyplot as plt
norm_sigma = 3.0

def gaussian(mu, sigma):
  return lambda x: 1 / np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2/(2*sigma**2))
fn_norm = gaussian(6.5, norm_sigma)
action_space = np.arange(14)
action_prob = fn_norm(action_space); action_prob /= action_prob.sum()

n_treasure = int(np.random.choice(action_space, 1, p=action_prob)[0])
print(action_prob, n_treasure)
plt.plot(action_space, action_prob)
plt.show()