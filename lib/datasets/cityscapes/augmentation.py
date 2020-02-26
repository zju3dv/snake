from lib.utils.data_utils import get_border, get_affine_transform, color_aug, blur_aug, small_aug
import numpy as np
import cv2
from lib.config import cfg


def augment(img, split, down_ratio, _data_rng, _eig_val, _eig_vec, mean, std, polys, boxes=None, label=None):
    # resize input
    height, width = img.shape[0], img.shape[1]
    center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    scale = max(img.shape[0], img.shape[1]) * 1.0
    scale = 800
    # __import__('ipdb').set_trace()
    # random crop and flip augmentation
    flipped = False
    if cfg.small_num > 0:
        img, polys, boxes, label = small_aug(img, polys, boxes, label, cfg.small_num)
    if split == 'train':
        scale = scale * np.random.choice(np.arange(0.6, 1.4, 0.1))
        seed = np.random.randint(0, len(polys))
        index = np.random.randint(0, len(polys[seed]))
        x = polys[seed][index]['bbox'][0] + (polys[seed][index]['bbox'][2] - 1) / 2
        y = polys[seed][index]['bbox'][1] + (polys[seed][index]['bbox'][3] - 1) / 2
        w_border = get_border(200, scale)
        h_border = get_border(200, scale)
        if (w_border == 0) or (h_border == 0):
            center[0] = x
            center[1] = y
        else:
            center[0] = np.random.randint(low=max(x-w_border, 0), high=min(x+w_border, width-1))
            center[1] = np.random.randint(low=max(y-h_border, 0), high=min(y+h_border, height-1))

        # flip augmentation
        if np.random.random() < 0.5:
            flipped = True
            img = img[:, ::-1, :]
            center[0] = width - center[0] - 1

    input_h, input_w = (800, 800)
    if split == 'val':
        center = np.array([1024, 512])
        scale = [2048, 1024]
        input_h, input_w = (1024, 2048)

    # print(center,scale)
    # print(flipped)
    # center = np.array([1272., 718.])
    # scale = 358.4
    # import ipdb; ipdb.set_trace()
    # center = np.array([1583., 306.])
    # print(center)
    # scale = 358.4
    # print(center, scale)
    trans_input = get_affine_transform(center, scale, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

    # color augmentation
    orig_img = inp.copy()
    inp = (inp.astype(np.float32) / 255.)
    if split == 'train':
        color_aug(_data_rng, inp, _eig_val, _eig_vec)
        # blur_aug(inp)

    # normalize the image
    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1)

    # resize output
    # if split == 'train':
    output_h = input_h // down_ratio
    output_w = input_w // down_ratio
    trans_output = get_affine_transform(center, scale, 0, [output_w, output_h])

    return orig_img, inp, trans_input, trans_output, input_h, input_w, output_h, output_w, flipped, center, scale, \
           polys, boxes, label
