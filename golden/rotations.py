# core
import itertools as it

# 3rd party
import numpy as np


def rot_x(alpha):
    return np.array([[1, 0, 0],
                     [0, np.cos(alpha), - np.sin(alpha)],
                     [0, np.sin(alpha), np.cos(alpha)]])


def rot_y(alpha):
    return np.array([[np.cos(alpha), 0, np.sin(alpha)],
                     [0, 1, 0],
                     [-np.sin(alpha), 0, np.cos(alpha)]])


def rot_z(alpha):
    return np.array([[np.cos(alpha), -np.sin(alpha), 0],
                     [np.sin(alpha), np.cos(alpha), 0],
                     [0, 0, 1]])


def rotate(X, rx, ry, rz):
    # X is an (N, 3) array
    # can be optimized to use a single rotation matrix
    X = np.dot(X, rot_x(rx))
    X = np.dot(X, rot_y(ry))
    X = np.dot(X, rot_z(rz))
    return X


def all_rots(n_per_dim):
    # n**3 rotations..
    xr, yr, zr = np.linspace(0, np.pi, n_per_dim), np.linspace(0, np.pi, n_per_dim), np.linspace(0, np.pi, n_per_dim)
    prod = it.product(xr, yr, zr)
    return list(prod)[:-1]  # last rotation is the starting position again, so it can be removed


def rotate_test():
    vec = np.array([1, 1, 1])
    print(np.dot(rot_x(np.pi/2), vec))
    print(np.dot(rot_y(np.pi/2), vec))
    print(np.dot(rot_z(np.pi/2), vec))

if __name__ == '__main__':
    rots = all_rots(5)
    print(len(rots))
    print(list(rots))