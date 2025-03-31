import numpy as np


def get_snapshot(filepath: str):
    data = np.genfromtxt(filepath, delimiter="\t")

    xs = np.array([data[-1, 1], data[-1, 1 + 3]])
    ys = np.array([data[-1, 2], data[-1, 2 + 3]])
    zs = np.array([data[-1, 3], data[-1, 3 + 3]])

    vxs = np.array([data[-1, 7], data[-1, 7 + 3]])
    vys = np.array([data[-1, 8], data[-1, 8 + 3]])
    vzs = np.array([data[-1, 9], data[-1, 9 + 3]])

    return xs, ys, zs, vxs, vys, vzs
