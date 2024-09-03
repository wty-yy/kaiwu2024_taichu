import numpy as np
import matplotlib.pyplot as plt


def bfs(grid, start, end):
    # 初始化
    rows, cols = len(grid), len(grid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    queue = [(start, [start])]  # 队列中的元素为(当前位置, 到当前位置的路径)

    # 方向向量：上，下，左，右
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        (x, y), path = queue.pop(0)
        if (x, y) == end:
            return path
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == 0:
                visited[nx][ny] = True
                queue.append(((nx, ny), path + [(nx, ny)]))
    return []


def plot_path(grid, path):
    (rows, cols) = grid.shape
    fig, ax = plt.subplots(dpi=300)

    for x in range(rows):
        for y in range(cols):
            newX, newY = x, y
            if grid[x][y] == 1:
                ax.fill_between([newX, newX + 1], newY, newY + 1, color='black')
            else:
                ax.fill_between([newX, newX + 1], newY, newY + 1, color='white')

    for (x, y) in path:
        newX, newY = x, y
        ax.fill_between([newX, newX + 1], newY, newY + 1, color='blue', alpha=0.5)

    start_x, start_y = path[0][0], path[0][1]
    end_x, end_y = path[-1][0], path[-1][1]
    ax.fill_between([start_x, start_x + 1], start_y, start_y + 1, color='red')
    ax.fill_between([end_x, end_x + 1], end_y, end_y + 1, color='green')

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    plt.gca().invert_yaxis
    plt.show()


if "name" == "__main__":
    grid = np.load("map.npy")
    path = bfs(grid, (29, 9), (11, 55))
    plot_path(grid, path)
