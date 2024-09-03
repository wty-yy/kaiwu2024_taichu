from diy.utils import is_iterable

def dfs_iterable_struct(x, func):
  """
  递归地对x的叶子结点作用func函数.
  """
  if not is_iterable(x):
    return func(x)
  if isinstance(x, dict):
    tmp = {}
    for k, v in x.items():
      tmp[k] = dfs_iterable_struct(v, func)
  if isinstance(x, list):
    tmp = []
    for i in x:
      tmp.append(dfs_iterable_struct(i, func))
  return tmp

if __name__ == '__main__':
  x = {'a': [2, 3,4 ], 'b': {'c': [1,2], 'd': 4}, 'e': 5}
  fn = lambda x: -x
  y = dfs_iterable_struct(x, type)
  print(x, y)
