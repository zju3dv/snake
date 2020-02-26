import torchvision
import os
from lib.utils.snake import snake_voc_utils, snake_config, visualize_utils
import cv2
import numpy as np
import math
from lib.utils import data_utils


class Dataset(torchvision.datasets.coco.CocoDetection):
    num_classes = 20
    CLASSES = snake_voc_utils.CLASSES

    def __init__(self, ann_file, data_root, split):
        super(Dataset, self).__init__(data_root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations and some gt are too small
        if split == 'val':
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ][:]

        # only select the cats you want
        # self.select_anno(2)

        self.data_root = data_root
        self.split = split

        self.json_category_id_to_contiguous_id = {
            v: i for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def select_anno(self, cat_id=1):
        ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            annos = self.coco.loadAnns(ann_ids)
            for ann in annos:
                if ann['category_id'] == cat_id:
                    ids.append(img_id)
                    break
        self.ids = ids

    def process_info(self, img_id):
        path = os.path.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
        return path

    def read_original_data(self, path):
        img = cv2.imread(path)
        return img


    def __getitem__(self, index):
        img_id = self.ids[index]
        # print(img_id)
        path = self.process_info(img_id)
        img = self.read_original_data(path)

        # height, width = img.shape[0], img.shape[1]
        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            snake_voc_utils.augment(
                img, self.split,
                snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
                snake_config.mean, snake_config.std
            )

        ret = {'inp': inp}

        meta = {'center': center, 'img_id': img_id, 'scale': scale, 'test': ''}
        ret.update({'meta': meta})

        return ret

    def __len__(self):
        return len(self.ids)
