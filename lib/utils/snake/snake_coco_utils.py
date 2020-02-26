from lib.utils.snake.snake_cityscapes_utils import *

input_scale = np.array([512, 512])


def augment(img, split, _data_rng, _eig_val, _eig_vec, mean, std, polys):
    # resize input
    height, width = img.shape[0], img.shape[1]
    center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    scale = max(height, width)
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    # random crop and flip augmentation
    flipped = False
    if split == 'train':
        scale = scale * np.random.uniform(0.6, 1.4)
        seed = np.random.randint(0, len(polys))
        index = np.random.randint(0, len(polys[seed][0]))
        x, y = polys[seed][0][index]
        center[0] = x
        border = scale[0] // 2 if scale[0] < width else width - scale[0] // 2
        center[0] = np.clip(center[0], a_min=border, a_max=width-border)
        center[1] = y
        border = scale[1] // 2 if scale[1] < height else height - scale[1] // 2
        center[1] = np.clip(center[1], a_min=border, a_max=height-border)

        # flip augmentation
        if np.random.random() < 0.5:
            flipped = True
            img = img[:, ::-1, :]
            center[0] = width - center[0] - 1

    input_w, input_h = input_scale
    if split != 'train':
        center = np.array([width // 2, height // 2])
        scale = np.array([width, height])
        x = 32
        input_w = (int(width / 1.) | (x - 1)) + 1
        input_h = (int(height / 1.) | (x - 1)) + 1
        scale = np.array([input_w, input_h])
        # input_w, input_h = (width + x - 1) // x * x, (height + x - 1) // x * x
        # input_w, input_h = int((width / 0.5 + x - 1) // x * x), int((height / 0.5 + x - 1) // x * x)
        # input_w, input_h = 512, 512

    trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

    # color augmentation
    orig_img = inp.copy()
    inp = (inp.astype(np.float32) / 255.)
    if split == 'train':
        data_utils.color_aug(_data_rng, inp, _eig_val, _eig_vec)
        # data_utils.blur_aug(inp)

    # normalize the image
    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1)

    output_h, output_w = input_h // snake_config.down_ratio, input_w // snake_config.down_ratio
    trans_output = data_utils.get_affine_transform(center, scale, 0, [output_w, output_h])
    inp_out_hw = (input_h, input_w, output_h, output_w)

    return orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw


