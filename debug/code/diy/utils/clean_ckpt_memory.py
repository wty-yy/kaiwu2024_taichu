from pathlib import Path
from kaiwudrl.common.config.config_control import CONFIG

def clean_ckpt_memory():
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
  if len(files) < 2: return
  for p in files[:-1]:  # just keep latest checkpoint
    p.unlink()