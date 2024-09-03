import tty
import sys
import termios

def wait_key():
  orig_settings = termios.tcgetattr(sys.stdin)
  tty.setcbreak(sys.stdin)
  x = sys.stdin.read(1)[0]
  termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)
  return x