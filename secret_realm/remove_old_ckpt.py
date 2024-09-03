import time
from pathlib import Path
path_ckpt = Path(__file__).parent / "train/backup_model"
min_to_keep = 10
sleep_time = 30 * 60  # second

while 1:
  print("Start check at:", time.strftime(r"%Y.%m.%d %H:%M:%S"))
  ckpts = sorted([x for x in path_ckpt.glob('*.zip') if x.is_file()],
                key=lambda x: int(x.stem.split('back_to_the_realm-diy-')[1].split('-')[0]))
  if len(ckpts) > min_to_keep:
    info = "Remove ckpts: "
    for p in ckpts[:-10]:
      info += p.stem + ' '
      p.unlink()
    print(info)
  else:
    print(f"ckpts number={len(ckpts)} < {min_to_keep}, nothing to do.")
  print(f"sleep {sleep_time}s...")
  time.sleep(sleep_time)