from kaiwu_agent.utils.common_func import create_cls, attached
import numpy as np

SampleData = create_cls("SampleData", a=None)
ObsData = create_cls("ObsData", feature=None)
ActData = create_cls("ActData", act=None)

def state2dict(s):
  assert len(s) == 250
  s = np.array(s, np.float32)
  s = {
    'pos': (np.argmax(s[1:65]), np.argmax(s[65:129])),
    'dist': {
      'end': s[129],
      'star': s[130:140],  # should use star_flag to check the star existed, if star is get
    },
    'map': {
      'wall': s[140:165].reshape(5, 5),
      'star': s[165:190].reshape(5, 5),
      'end': s[190:215].reshape(5, 5),
    },
    'memory': s[215:240].reshape(5, 5),
    'star_flag': s[240:250]
  }
  types = {
    'pos': np.int32,
    'dist': np.int32,
    'map': bool,
    'memory': np.float32,
    'star_flag': bool 
  }
  for k, v in s.items():
    t = types[k]
    if not isinstance(v, dict):
      s[k] = np.array(v, t)
    else:
      for kk in v:
        v[kk] = np.array(v[kk], types[k])
  return s

@attached
def observation_process(raw_obs):
  return ObsData(feature=state2dict(raw_obs))

@attached
def action_process(act_data):
    return act_data.act

@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]
