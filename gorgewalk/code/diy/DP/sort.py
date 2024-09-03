from bfs import bfs
from bfs import plot_path
import numpy as np
treasure = [(19, 14),(9, 28),(9, 44),(42, 45),(32, 23),(49, 56),(35, 58),(23, 55),(41, 33),(54, 41)]
map = np.load("map.npy")


def permute(lst):
    if len(lst) == 0:
        return [[]]
    result = []
    for i in range(len(lst)):
        element = lst[i]
        rest = lst[:i] + lst[i + 1:]
        for p in permute(rest):
            result.append([element] + p)
    return result


def sort(visiable_treasure: list):
    per = permute(visiable_treasure)
    start = (29, 9)
    end = (11, 55)
    shortest_path = 0
    for i in per:
        path = [start]
        for j in i:
            path += bfs(map, path[-1], treasure[j])
            if len(path) >= shortest_path and shortest_path != 0:
                # prefix = i[:i.index(j) + 1]
                # for k in per:
                #     if k[:i.index(j) + 1] == prefix:
                #         per.remove(k)
                break
        path += bfs(map, path[-1], end)
        # per.remove(i)
        if len(path) < shortest_path or shortest_path == 0:
            shortest_path = len(path)
            best_path = path
    return best_path


visiable_treasure = [1,3,6,8]
plot_path(map, sort(visiable_treasure))
