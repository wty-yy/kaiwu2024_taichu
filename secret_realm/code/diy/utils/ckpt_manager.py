from pathlib import Path
from kaiwudrl.common.config.config_control import CONFIG

def clean_ckpt_memory(path_name="restore", logger=None, min_num_model=10):
  """
  Remove old checkpoints generate by autosave (save frequency=CONFIG.dump_model_freq),
  you can find autosave code at
  `kaiwudrl.common.algorithms.standard_model_wrapper_pytorch.StandardModelWrapperPytorch.after_train()`

  Usage: Call this function in `agent.learn(...)` or `train_workflow.workflow`, 
  recommend in `agent.learn(...)` since it is a single process,
  add small delay as you like~~~
  """
  if path_name == 'restore':
    path_tmp_dir = Path(f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/")
  if path_name == 'user':
    path_tmp_dir = Path(f"{CONFIG.user_ckpt_dir}/{CONFIG.app}_{CONFIG.algo}/")
  files = sorted(list(path_tmp_dir.glob('model.ckpt-*')), key=lambda x: int(str(x).rsplit('.', 1)[0].rsplit('-', 1)[1]))
  if len(files) > min_num_model:
    info = f"Remove old ckpts ({path_tmp_dir}): "
    for p in files[:-min_num_model]:  # just keep latest checkpoint
      p.unlink()
      info += f"{p.stem} "
    if logger is not None:
      logger.info(info)

def get_latest_ckpt_path(path=Path(f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/")):
  files = sorted(list(path.glob('model.ckpt-*')), key=lambda x: int(str(x).rsplit('.', 1)[0].rsplit('-', 1)[1]))
  if len(files) == 0: return None
  return str(files[-1])
