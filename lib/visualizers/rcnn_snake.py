from lib.utils import img_utils, data_utils
from lib.utils.snake import snake_config
import matplotlib.pyplot as plt
import numpy as np
import torch
import pycocotools.coco as coco
from lib.config import cfg
from lib.datasets.dataset_catalog import DatasetCatalog
import os
import cv2
from itertools import cycle

mean = snake_config.mean
std = snake_config.std


class Visualizer:
    def __init__(self):
        self.coco = None

    def visualize_ex(self, output, batch):
        inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0))

        detection = output['detection']
        score = detection[:, 4].detach().cpu().numpy()
        label = detection[:, 5].detach().cpu().numpy().astype(int)
        cp_ind = output['cp_ind'].detach().cpu().numpy().astype(int)
        py = output['py'][-1].detach().cpu().numpy() * snake_config.down_ratio

        if len(py) == 0:
            return

        ct_ind = np.unique(cp_ind)
        score = score[ct_ind]
        label = label[ct_ind]
        ind_group = [np.argwhere(ct_ind[i] == cp_ind).ravel() for i in range(len(ct_ind))]
        py = [[py[ind] for ind in inds] for inds in ind_group]

        fig, ax = plt.subplots(1, figsize=(20, 10))
        fig.tight_layout()
        ax.axis('off')
        ax.imshow(inp)

        colors = np.array([
            [31, 119, 180],
            [255, 127, 14],
            [46, 160, 44],
            [214, 40, 39],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [126, 126, 126],
            [188, 189, 32],
            [26, 190, 207]
        ]) / 255.
        # colors = cycle(colors)
        for i in range(len(py)):
            color = colors[np.random.randint(len(colors))]
            # color = next(colors).tolist()
            for poly in py[i]:
                poly = np.append(poly, [poly[0]], axis=0)
                ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=3)

        plt.show()

    def visualize_training_box(self, output, batch):
        inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0))
        box = output['detection'][:, :4].detach().cpu().numpy() * snake_config.down_ratio
        # box = output['cp_box'][:, :4].detach().cpu().numpy() * snake_config.down_ratio

        _, ax = plt.subplots(1)
        ax.imshow(inp)
        n = len(box)
        for i in range(n):
            x_min, y_min, x_max, y_max = box[i]
            ax.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min])
        plt.show()

    def visualize(self, output, batch):
        self.visualize_ex(output, batch)
        # self.visualize_training_box(output, batch)

