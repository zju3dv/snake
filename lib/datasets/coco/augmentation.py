from lib.utils.data_utils import get_border, get_affine_transform, color_aug
import numpy as np
import cv2


def augment(img, split, down_ratio, _data_rng, _eig_val, _eig_vec, mean, std):
    # resize input
    height, width = img.shape[0], img.shape[1]
    center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    scale = max(img.shape[0], img.shape[1]) * 1.0

    # random crop and flip augmentation
    flipped = False
    if split == 'train':
        scale = scale * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = get_border(128, img.shape[1])
        h_border = get_border(128, img.shape[0])
        center[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        center[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)

        # flip augmentation
        if np.random.random() < 0.5:
            flipped = True
            img = img[:, ::-1, :]
            center[0] = width - center[0] - 1

    input_h, input_w = (512, 512)
    trans_input = get_affine_transform(center, scale, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

    # color augmentation
    orig_img = inp.copy()
    inp = (inp.astype(np.float32) / 255.)
    if split == 'train':
        color_aug(_data_rng, inp, _eig_val, _eig_vec)

    # normalize the image
    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1)

    # resize output
    output_h = input_h // down_ratio
    output_w = input_w // down_ratio
    trans_output = get_affine_transform(center, scale, 0, [output_w, output_h])

    return orig_img, inp, trans_input, trans_output, input_h, input_w, output_h, output_w, flipped
