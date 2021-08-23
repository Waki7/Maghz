from typing import *
from shapely.geometry import Polygon

import matplotlib.pyplot as plt
import numpy as np
import torch


def _rotation(pts: np.ndarray, theta: float) -> np.ndarray:
    '''
    rotation along x axis
    '''
    center = np.mean(pts, axis=0)
    unit_pts = pts - center
    r = np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]])
    unit_pts = unit_pts @ r
    return unit_pts + center


def plot(ax, img, pts_list: List[np.ndarray], title):
    ax.imshow(img, cmap="gray")
    ax.set_title(title)

    for xy in pts_list:
        # ax.scatter(x, y, c="r")
        ax.plot(xy[:, 0], xy[:, 1], c="r")


fig, ax = plt.subplots(1, 3, figsize=(12, 4))
img = np.zeros((200, 200))
pts1 = np.array([
    [40, 40], [40, 60], [60, 60], [60, 40]
])
pts2 = _rotation(np.array([
    [40, 30], [40, 60], [60, 60], [60, 30]
]), .25 * np.pi)
print(pts2)
p1 = Polygon(pts1)
p2 = Polygon(pts2)
print(p1.intersection(p2).area)
print(pts1[0, :].shape)

t1 = torch.Tensor(pts1)
t2 = torch.Tensor(pts2)
intersect = torch.stack([
    torch.min(t1[0], t2[0]),
    torch.min(t1[1], t2[1]),
    torch.min(t1[2], t2[2]),
    torch.min(t1[3], t2[3]),
]).numpy()

plot(ax[0], img, [pts1, pts2, intersect], "example (with spaceship)")

print(intersect.shape)
pi = Polygon(intersect)
print(pi.area)
plt.show()
