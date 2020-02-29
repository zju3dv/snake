import torch.utils.data as data
import glob
import os
import cv2
import numpy as np
from lib.utils.snake import snake_config
from lib.utils import data_utils


class Dataset(data.Dataset):
    def __init__(self, data_root):
        super(Dataset, self).__init__()

        self.imgs = glob.glob(os.path.join(data_root, '*/*.png'))

    def normalize_image(self, inp):
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - snake_config.mean) / snake_config.std
        inp = inp.transpose(2, 0, 1)
        return inp

    def __getitem__(self, index):
        img = self.imgs[index]
        img_id = os.path.basename(img).replace('_leftImg8bit.png', '')
        img = cv2.imread(img)

        width, height = 2048, 1024
        center = np.array([width // 2, height // 2])
        scale = np.array([width, height])
        # input_w, input_h = width, height
        input_w, input_h = int((width / 0.85 + 31) // 32 * 32), int((height / 0.85 + 31) // 32 * 32)
        trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

        inp = self.normalize_image(inp)
        ret = {'inp': inp}
        meta = {'center': center, 'scale': scale, 'test': '', 'img_id': img_id, 'ann': ''}
        ret.update({'meta': meta})

        return ret

    def __len__(self):
        return len(self.imgs)

