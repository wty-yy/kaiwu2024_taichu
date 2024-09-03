import traceback

debug_colors = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan']
debug_idxs = 0

def colorstr(*input):
  # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
  *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
  colors = {
    'black': '\033[30m',  # basic colors
    'red': '\033[31m',
    'green': '\033[32m',
    'yellow': '\033[33m',
    'blue': '\033[34m',
    'magenta': '\033[35m',
    'cyan': '\033[36m',
    'white': '\033[37m',
    'bright_black': '\033[90m',  # bright colors
    'bright_red': '\033[91m',
    'bright_green': '\033[92m',
    'bright_yellow': '\033[93m',
    'bright_blue': '\033[94m',
    'bright_magenta': '\033[95m',
    'bright_cyan': '\033[96m',
    'bright_white': '\033[97m',
    'end': '\033[0m',  # misc
    'bold': '\033[1m',
    'underline': '\033[4m'}
  return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def show_debug(info, ptext="[DEBUG]", render_cfg=None, verbose_depth=1):
  def from_pos(depth=1):
    tmp = frame_summary[-1-depth]
    return f"<{tmp.filename}, line {tmp.lineno}> in {tmp.name} "

  frame_summary = traceback.extract_stack()
  verbose_depth = min(verbose_depth, len(frame_summary)-1)
  if not isinstance(info, tuple):
     info = (info,)
  if render_cfg is None:
    global debug_idxs
    render_cfg = (debug_colors[debug_idxs], 'bold')
    debug_idxs = (debug_idxs + 1) % len(debug_colors)

  for i in range(verbose_depth):
    tmp = ' FROM ' if i == 0 else ' IN   '
    print(colorstr(*render_cfg, f"{ptext+tmp+from_pos(i+1):+<120}"))
  print(*info)
  print(colorstr(*render_cfg, f"{ptext+' END  '+from_pos(1):-<120}"))

if __name__ == '__main__':
  print(colorstr('red', "hi"))
  def foo():
    show_debug("hi", verbose_depth=10)
  foo()
