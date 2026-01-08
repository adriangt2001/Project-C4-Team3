from collections.abc import Iterable
from typing import Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


def projective2img(x):
    """
    Take a 3D homogenous/projective coordinate and obtain the 2D (image) euclidean equivalent
    """
    assert x.shape[0] == 3, f'`x` shape {x.shape} expected (3,...)'
    return x[:2, ...] / x[2][np.newaxis]


def img2projective(x):
    """
    Take a 2D image coordinate and add the 3th dimension of homogenous/projective coordinate
    """
    assert x.shape[0] == 2, f'`x` shape {x.shape} expected (2,...) '
    return np.vstack((x, np.ones_like(x[0, ...])))


def homogeneous2euclidean(x):
    """
    Take a 4D homogenous coordinate and normalize to obtain the 3D euclidan equivalent
    Take a 3D homogenous/projective coordinate and obtain the 2D (image) euclidean equivalent
    """
    assert x.shape[0] == 4, f'`x` shape {x.shape} expected (4,...)'
    return x[:3, ...] / x[3][np.newaxis]


def euclidean2homogeneous(x):
    """
    Take a 3D euclidean coordinate and add the 4th dimension of homogenous cordinate
    """
    assert x.shape[0] == 3, f'`x` shape {x.shape} expected (3,...) '
    return np.vstack((x, np.ones_like(x[0, ...])))


def line_draw(line, canv, size):
    def get_y(t):
        return -(line[0] * t + line[2]) / line[1]

    def get_x(t):
        return -(line[1] * t + line[2]) / line[0]

    w, h = size

    if line[0] != 0 and abs(get_x(0) - get_x(w)) < w:
        beg = (get_x(0), 0)
        end = (get_x(h), h)
    else:
        beg = (0, get_y(0))
        end = (w, get_y(w))
    canv.line([beg, end], width=4)


def plot_img(img, do_not_use=[0]):
    plt.figure(do_not_use[0])
    do_not_use[0] += 1
    plt.imshow(img)


def optical_center(P):
    u, s, vh = np.linalg.svd(P)
    o = vh[:, -1]
    o = o[:3] / o[3]

    return o


def view_direction(P, x):
    v, resid, rank, s = np.linalg.lstsq(P[:, :3], x, rcond=None)
    return v


def draw_points(points, ax=None, **plot_kwargs):
    # Creating figure
    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection="3d")
    # Creating plot
    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], **plot_kwargs)


def draw_lines(points, ax: Optional[Axes3D] = None, **line_kwargs):
    """
    :param points: shape (num_points, 2, 3)
    :param ax:
    :return:
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection="3d")
    assert isinstance(ax, Axes3D)

    for l in points:
        p1, p2 = l[0], l[1]
        ax.plot(xs=[p1[0], p2[0]], ys=[p1[1], p2[1]], zs=[p1[2], p2[2]], **line_kwargs)


def get_camera_frame_points(P, w, h, scale, ax=None):
    """
    :param P: Camera matrix
    :param w: Width
    :param h: Height
    :param scale: Scale
    :return:
    """
    o = optical_center(P)
    p1 = o + view_direction(P, np.array([0, 0, 1])) * scale
    p2 = o + view_direction(P, np.array([w, 0, 1])) * scale
    p3 = o + view_direction(P, np.array([w, h, 1])) * scale
    p4 = o + view_direction(P, np.array([0, h, 1])) * scale

    points = np.array([[o, p1]])
    points = np.vstack((points, np.array([[o, p2]])))
    points = np.vstack((points, np.array([[o, p3]])))
    points = np.vstack((points, np.array([[o, p4]])))
    points = np.vstack((points, np.array([[p1, p2]])))
    points = np.vstack((points, np.array([[p2, p3]])))
    points = np.vstack((points, np.array([[p3, p4]])))
    points = np.vstack((points, np.array([[p4, p1]])))
    return points


def plot_camera(P, w, h, scale, ax=None, **plot_kwargs):
    """
    :param P: Camera matrix
    :param w: Width
    :param h: Height
    :param scale: Scale
    :param ax: matplotlib axis
    :return:
    """
    points = get_camera_frame_points(P, w, h, scale)
    draw_lines(points, ax=ax, **plot_kwargs)

def line_draw(line, canv, size):
    def get_y(t):
        return -(line[0] * t + line[2]) / line[1]

    def get_x(t):
        return -(line[1] * t + line[2]) / line[0]

    w, h = size

    if line[0] != 0 and abs(get_x(0) - get_x(w)) < w:
        beg = (get_x(0), 0)
        end = (get_x(h), h)
    else:
        beg = (0, get_y(0))
        end = (w, get_y(w))
    canv.line([beg, end], width=4)


def plot_img(img, do_not_use=[0]):
    plt.figure(do_not_use[0])
    do_not_use[0] += 1
    plt.imshow(img)
