import os
from pathlib import Path
PATH_DEBUG_DIR = Path(__file__).parents[2] / "debug"
PATH_DEBUG_DIR.mkdir(exist_ok=True)

class ProcessPrinter:
  def __init__(self):
    path = PATH_DEBUG_DIR / f"{os.getpid()}.log"
    idx = 2
    while path.exists():
      path = path.with_stem(path.stem.split('_')[0] + f'_{idx}')
      idx += 1
    self.path = path
  
  def __call__(self, s: str, newline=True):
    if newline and s[-1] != '\n': s += '\n'
    with open(self.path, 'a') as file:
      file.write(s)

def show_hi():
  process_printer = ProcessPrinter()
  process_printer('hi')

if __name__ == '__main__':
  show_hi()
