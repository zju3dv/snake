import glob
import os
import json
import numpy as np
from lib.utils import data_utils
import cv2
from lib.utils.snake import snake_config
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import polygonize
from scipy import ndimage
import math

# Globals ----------------------------------------------------------------------
COCO_LABELS = {24: 1,
               26: 2,
               27: 3,
               25: 4,
               33: 5,
               32: 6,
               28: 7,
               31: 8}

# Label number to name and color
INSTANCE_LABELS = {26: {'name': 'car', 'color': [0, 0, 142]},
                   24: {'name': 'person', 'color': [220, 20, 60]},
                   25: {'name': 'rider', 'color': [255, 0, 0]},
                   32: {'name': 'motorcycle', 'color': [0, 0, 230]},
                   33: {'name': 'bicycle', 'color': [119, 11, 32]},
                   27: {'name': 'truck', 'color': [0, 0, 70]},
                   28: {'name': 'bus', 'color': [0, 60, 100]},
                   31: {'name': 'train', 'color': [0, 80, 100]}}

# Label name to number
LABEL_DICT = {'car': 26, 'person': 24, 'rider': 25, 'motorcycle': 32,
              'bicycle': 33, 'truck': 27, 'bus': 28, 'train': 31}
# LABEL_DICT = {'bicycle': 33}

# Label name to contiguous number
JSON_DICT = dict(car=0, person=1, rider=2, motorcycle=3, bicycle=4, truck=5, bus=6, train=7)
# JSON_DICT = dict(bicycle=0)
# Contiguous number to name
NUMBER_DICT = {0: 'car', 1: 'person', 2: 'rider', 3: 'motorcycle',
               4: 'bicycle', 5: 'truck', 6: 'bus', 7: 'train'}
# NUMBER_DICT = {0:'bicycle'}
# Array of keys
KEYS = np.array([[26000, 26999], [24000, 24999], [25000, 25999],
                 [32000, 32999], [33000, 33999], [27000, 27999],
                 [28000, 28999], [31000, 31999]])

NUM_CLASS = {'person': 17914, 'rider': 1755, 'car': 26944, 'truck': 482,
             'bus': 379, 'train': 168, 'motorcycle': 735, 'bicycle': 3658}

# ------------------------------------------------------------------------------


def continuous_label_to_cityscapes_label(labels):
    return np.array([LABEL_DICT[NUMBER_DICT[label]] for label in labels])


def read_dataset(ann_files):
    if not isinstance(ann_files, tuple):
        ann_files = (ann_files,)

    ann_file = []
    for ann_file_dir in ann_files:
        ann_file += glob.glob(os.path.join(ann_file_dir, '*/*.json'))

    ann_filter = []
    for fname in ann_file:
        with open(fname, 'r') as f:
            ann = json.load(f)
            examples = []
            for instance in ann:
                instance_label = instance['label']
                if instance_label not in LABEL_DICT:
                    continue
                examples.append(instance)
            if len(examples) > 0:
                ann_filter.append(fname)
    return ann_filter


def process_info(fname, data_root):
    with open(fname, 'r') as f:
        ann = json.load(f)

    examples = []

    for instance in ann:
        instance_label = instance['label']
        if instance_label not in LABEL_DICT:
            continue
        examples.append(instance)

    img_path = os.path.join(data_root, '/'.join(ann[0]['img_path'].split('/')[-3:]))
    img_id = ann[0]['image_id']

    return examples, img_path, img_id


def xywh_to_xyxy(boxes):
    """
    boxes: [[x, y, w, h]]
    """
    boxes = np.array(boxes)
    if len(boxes) == 0:
        return boxes
    x, y, w, h = np.split(boxes, 4, axis=1)
    x_max = x + w
    y_max = y + h
    return np.concatenate([x, y, x_max, y_max], axis=1)


def augment(img, split, _data_rng, _eig_val, _eig_vec, mean, std, polys):
    # resize input
    height, width = img.shape[0], img.shape[1]
    center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    scale = snake_config.scale
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    # random crop and flip augmentation
    flipped = False
    if split == 'train':
        scale = scale * np.random.uniform(0.4, 1.6)
        seed = np.random.randint(0, len(polys))
        index = np.random.randint(0, len(polys[seed]))
        poly = polys[seed][index]['poly']
        center[0], center[1] = poly[np.random.randint(len(poly))]
        border = scale[0] // 2 if scale[0] < width else width - scale[0] // 2
        center[0] = np.clip(center[0], a_min=border, a_max=width-border)
        border = scale[1] // 2 if scale[1] < height else height - scale[1] // 2
        center[1] = np.clip(center[1], a_min=border, a_max=height-border)

        # flip augmentation
        if np.random.random() < 0.5:
            flipped = True
            img = img[:, ::-1, :]
            center[0] = width - center[0] - 1

    input_w, input_h = snake_config.input_w, snake_config.input_h
    if split != 'train':
        center = np.array([width // 2, height // 2])
        scale = np.array([width, height])
        # input_w, input_h = width, height
        input_w, input_h = int((width / 0.85 + 31) // 32 * 32), int((height / 0.85 + 31) // 32 * 32)

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


def transform_bbox(bbox, trans_output, output_h, output_w):
    bbox = data_utils.affine_transform(bbox.reshape(-1, 2), trans_output).ravel()
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
    return bbox


def handle_break_point(poly, axis, number, outside_border):
    if len(poly) == 0:
        return []

    if len(poly[outside_border(poly[:, axis], number)]) == len(poly):
        return []

    break_points = np.argwhere(
        outside_border(poly[:-1, axis], number) != outside_border(poly[1:, axis], number)).ravel()
    if len(break_points) == 0:
        return poly

    new_poly = []
    if not outside_border(poly[break_points[0], axis], number):
        new_poly.append(poly[:break_points[0]])

    for i in range(len(break_points)):
        current_poly = poly[break_points[i]]
        next_poly = poly[break_points[i] + 1]
        mid_poly = current_poly + (next_poly - current_poly) * (number - current_poly[axis]) / (next_poly[axis] - current_poly[axis])

        if outside_border(poly[break_points[i], axis], number):
            if mid_poly[axis] != next_poly[axis]:
                new_poly.append([mid_poly])
            next_point = len(poly) if i == (len(break_points) - 1) else break_points[i + 1]
            new_poly.append(poly[break_points[i] + 1:next_point])
        else:
            new_poly.append([poly[break_points[i]]])
            if mid_poly[axis] != current_poly[axis]:
                new_poly.append([mid_poly])

    if outside_border(poly[-1, axis], number) != outside_border(poly[0, axis], number):
        current_poly = poly[-1]
        next_poly = poly[0]
        mid_poly = current_poly + (next_poly - current_poly) * (number - current_poly[axis]) / (next_poly[axis] - current_poly[axis])
        new_poly.append([mid_poly])

    return np.concatenate(new_poly)


def transform_polys(polys, trans_output, output_h, output_w):
    new_polys = []
    for i in range(len(polys)):
        poly = polys[i]
        poly = data_utils.affine_transform(poly, trans_output)
        poly = handle_break_point(poly, 0, 0, lambda x, y: x < y)
        poly = handle_break_point(poly, 0, output_w, lambda x, y: x >= y)
        poly = handle_break_point(poly, 1, 0, lambda x, y: x < y)
        poly = handle_break_point(poly, 1, output_h, lambda x, y: x >= y)
        if len(poly) == 0:
            continue
        if len(np.unique(poly, axis=0)) <= 2:
            continue
        new_polys.append(poly)
    return new_polys


def get_valid_shape_poly(poly):
    """a poly may be self-intersected"""
    shape_poly = Polygon(poly)
    if shape_poly.is_valid:
        if shape_poly.area < 5:
            return []
        else:
            return [shape_poly]

    # self-intersected situation
    linering = shape_poly.exterior

    # disassemble polygons from multiple line strings
    mls = linering.intersection(linering)
    # assemble polygons from multiple line strings
    polygons = polygonize(mls)
    multi_shape_poly = MultiPolygon(polygons)
    shape_polys = []
    for shape_poly in multi_shape_poly:
        if shape_poly.area < 5:
            continue
        shape_polys.append(shape_poly)
    return shape_polys


def get_valid_polys(polys):
    """create shape_polys and filter polys"""
    # convert polygons into shape_poly
    shape_polys = []
    for poly in polys:
        shape_polys.extend(get_valid_shape_poly(poly))

    # remove polys being contained
    n = len(shape_polys)
    relation = np.zeros([n, n], dtype=np.bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            relation[i, j] = shape_polys[j].contains(shape_polys[i])

    relation = np.any(relation, axis=1)
    shape_polys = [shape_polys[i] for i, shape_poly in enumerate(shape_polys) if not relation[i]]
    polys = [np.array(shape_poly.exterior.coords)[::-1]
             if shape_poly.exterior.is_ccw else np.array(shape_poly.exterior.coords)
             for shape_poly in shape_polys]
    return polys


def filter_tiny_polys(polys):
    polys_ = []
    for poly in polys:
        x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
        x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
        if x_max - x_min >= 1 and y_max - y_min >= 1:
            polys_.append(poly)
    return [poly for poly in polys_ if Polygon(poly).area > 5]


def get_cw_polys(polys):
    return [poly[::-1] if Polygon(poly).exterior.is_ccw else poly for poly in polys]


def polygon_to_mask(poly, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(poly).astype(int)], 1)
    return mask


def get_inner_center(mask, mask_ct, h_int, w_int):
    mask_ct_int = np.round(mask_ct).astype(np.int32)
    if snake_config.box_center and mask[mask_ct_int[1], mask_ct_int[0]] == 1:
        ct = mask_ct_int
    else:
        dt = ndimage.distance_transform_edt(mask)
        dt_max = ndimage.maximum_filter(dt, footprint=np.ones([3, 3]))
        mask = (dt_max == dt) * mask

        radius = np.sqrt(h_int * h_int + w_int * w_int) / 6
        xy = np.argwhere(mask)[:, [1, 0]]
        dt = dt[xy[:, 1], xy[:, 0]]
        ct_distance = np.sqrt(np.power(xy - mask_ct, 2).sum(1))
        inlier = ct_distance < radius
        if snake_config.center_scope and len(np.argwhere(inlier)) > 0:
            xy = xy[inlier]
            dt = dt[inlier]
            xy = xy[np.argwhere(dt == np.max(dt)).ravel()]
            ct = xy[np.power(xy - mask_ct, 2).sum(1).argmin()]
        else:
            xy = np.argwhere(mask)[:, [1, 0]]
            ct = xy[np.power(xy - mask_ct, 2).sum(1).argmin()]
    return mask_ct_int


def prepare_ct_off_mask(mask_poly, mask_ct, h_int, w_int):
    mask_poly = mask_poly + 1
    mask_ct = mask_ct + 1
    mask = polygon_to_mask(mask_poly, h_int + 2, w_int + 2)
    ct = get_inner_center(mask, mask_ct, h_int, w_int) - 1
    mask = mask[1:-1, 1:-1]
    xy = np.argwhere(mask)[:, [1, 0]]
    off = ct - xy
    return ct, off, xy


def get_extreme_points(pts):
    l, t = min(pts[:, 0]), min(pts[:, 1])
    r, b = max(pts[:, 0]), max(pts[:, 1])
    # 3 degrees
    thresh = 0.02
    w = r - l + 1
    h = b - t + 1

    t_idx = np.argmin(pts[:, 1])
    t_idxs = [t_idx]
    tmp = (t_idx + 1) % pts.shape[0]
    while tmp != t_idx and pts[tmp, 1] - pts[t_idx, 1] <= thresh * h:
        t_idxs.append(tmp)
        tmp = (tmp + 1) % pts.shape[0]
    tmp = (t_idx - 1) % pts.shape[0]
    while tmp != t_idx and pts[tmp, 1] - pts[t_idx, 1] <= thresh * h:
        t_idxs.append(tmp)
        tmp = (tmp - 1) % pts.shape[0]
    tt = [(max(pts[t_idxs, 0]) + min(pts[t_idxs, 0])) / 2, t]

    b_idx = np.argmax(pts[:, 1])
    b_idxs = [b_idx]
    tmp = (b_idx + 1) % pts.shape[0]
    while tmp != b_idx and pts[b_idx, 1] - pts[tmp, 1] <= thresh * h:
        b_idxs.append(tmp)
        tmp = (tmp + 1) % pts.shape[0]
    tmp = (b_idx - 1) % pts.shape[0]
    while tmp != b_idx and pts[b_idx, 1] - pts[tmp, 1] <= thresh * h:
        b_idxs.append(tmp)
        tmp = (tmp - 1) % pts.shape[0]
    bb = [(max(pts[b_idxs, 0]) + min(pts[b_idxs, 0])) / 2, b]

    l_idx = np.argmin(pts[:, 0])
    l_idxs = [l_idx]
    tmp = (l_idx + 1) % pts.shape[0]
    while tmp != l_idx and pts[tmp, 0] - pts[l_idx, 0] <= thresh * w:
        l_idxs.append(tmp)
        tmp = (tmp + 1) % pts.shape[0]
    tmp = (l_idx - 1) % pts.shape[0]
    while tmp != l_idx and pts[tmp, 0] - pts[l_idx, 0] <= thresh * w:
        l_idxs.append(tmp)
        tmp = (tmp - 1) % pts.shape[0]
    ll = [l, (max(pts[l_idxs, 1]) + min(pts[l_idxs, 1])) / 2]

    r_idx = np.argmax(pts[:, 0])
    r_idxs = [r_idx]
    tmp = (r_idx + 1) % pts.shape[0]
    while tmp != r_idx and pts[r_idx, 0] - pts[tmp, 0] <= thresh * w:
        r_idxs.append(tmp)
        tmp = (tmp + 1) % pts.shape[0]
    tmp = (r_idx - 1) % pts.shape[0]
    while tmp != r_idx and pts[r_idx, 0] - pts[tmp, 0] <= thresh * w:
        r_idxs.append(tmp)
        tmp = (tmp - 1) % pts.shape[0]
    rr = [r, (max(pts[r_idxs, 1]) + min(pts[r_idxs, 1])) / 2]

    return np.array([tt, ll, bb, rr])


def get_quadrangle(box):
    x_min, y_min, x_max, y_max = box
    quadrangle = [
        [(x_min + x_max) / 2., y_min],
        [x_min, (y_min + y_max) / 2.],
        [(x_min + x_max) / 2., y_max],
        [x_max, (y_min + y_max) / 2.]
    ]
    return np.array(quadrangle)


def get_box(box):
    x_min, y_min, x_max, y_max = box
    box = [
        [(x_min + x_max) / 2., y_min],
        [x_min, y_min],
        [x_min, (y_min + y_max) / 2.],
        [x_min, y_max],
        [(x_min + x_max) / 2., y_max],
        [x_max, y_max],
        [x_max, (y_min + y_max) / 2.],
        [x_max, y_min]
    ]
    return np.array(box)


def get_init(box):
    if snake_config.init == 'quadrangle':
        return get_quadrangle(box)
    else:
        return get_box(box)


def get_octagon(ex):
    w, h = ex[3][0] - ex[1][0], ex[2][1] - ex[0][1]
    t, l, b, r = ex[0][1], ex[1][0], ex[2][1], ex[3][0]
    x = 8.0
    octagon = [
        ex[0][0], ex[0][1],
        max(ex[0][0] - w / x, l), ex[0][1],
        ex[1][0], max(ex[1][1] - h / x, t),
        ex[1][0], ex[1][1],
        ex[1][0], min(ex[1][1] + h / x, b),
        max(ex[2][0] - w / x, l), ex[2][1],
        ex[2][0], ex[2][1],
        min(ex[2][0] + w / x, r), ex[2][1],
        ex[3][0], min(ex[3][1] + h / x, b),
        ex[3][0], ex[3][1],
        ex[3][0], max(ex[3][1] - h / x, t),
        min(ex[0][0] + w / x, r), ex[0][1],
    ]
    return np.array(octagon).reshape(-1, 2)


def get_ellipse(box, cp_num):
    pointsnp = np.zeros(shape=(cp_num, 2), dtype=np.float32)
    for i in range(cp_num):
        theta = 1.0 * i / cp_num * 2 * np.pi
        x = np.cos(theta)
        y = -np.sin(theta)
        pointsnp[i, 0] = x
        pointsnp[i, 1] = y
    pointsnp /= 2.
    pointsnp += 0.5
    w, h = box[2] - box[0], box[3] - box[1]
    pointsnp *= np.array([w, h])
    pointsnp = pointsnp + np.array([box[0], box[1]])
    return pointsnp


def uniform_sample_init(poly):
    polys = []
    ind = np.array(list(range(0, len(poly), len(poly)//4)))
    next_ind = np.roll(ind, shift=-1)
    for i in range(len(ind)):
        poly_ = poly[ind[i]:ind[i]+len(poly)//4]
        poly_ = np.append(poly_, [poly[next_ind[i]]], axis=0)
        poly_ = uniform_sample_segment(poly_, snake_config.init_poly_num // 4)
        polys.append(poly_)
    return np.concatenate(polys)


def uniformsample(pgtnp_px2, newpnum):
    pnum, cnum = pgtnp_px2.shape
    assert cnum == 2

    idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
    pgtnext_px2 = pgtnp_px2[idxnext_p]
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
    edgeidxsort_p = np.argsort(edgelen_p)

    # two cases
    # we need to remove gt points
    # we simply remove shortest paths
    if pnum > newpnum:
        edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
        edgeidxsort_k = np.sort(edgeidxkeep_k)
        pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
        assert pgtnp_kx2.shape[0] == newpnum
        return pgtnp_kx2
    # we need to add gt points
    # we simply add it uniformly
    else:
        edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
        for i in range(pnum):
            if edgenum[i] == 0:
                edgenum[i] = 1

        # after round, it may has 1 or 2 mismatch
        edgenumsum = np.sum(edgenum)
        if edgenumsum != newpnum:

            if edgenumsum > newpnum:

                id = -1
                passnum = edgenumsum - newpnum
                while passnum > 0:
                    edgeid = edgeidxsort_p[id]
                    if edgenum[edgeid] > passnum:
                        edgenum[edgeid] -= passnum
                        passnum -= passnum
                    else:
                        passnum -= edgenum[edgeid] - 1
                        edgenum[edgeid] -= edgenum[edgeid] - 1
                        id -= 1
            else:
                id = -1
                edgeid = edgeidxsort_p[id]
                edgenum[edgeid] += newpnum - edgenumsum

        assert np.sum(edgenum) == newpnum

        psample = []
        for i in range(pnum):
            pb_1x2 = pgtnp_px2[i:i + 1]
            pe_1x2 = pgtnext_px2[i:i + 1]

            pnewnum = edgenum[i]
            wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i]

            pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
            psample.append(pmids)

        psamplenp = np.concatenate(psample, axis=0)
        return psamplenp


def uniform_sample_segment(pgtnp_px2, newpnum):
    pnum, cnum = pgtnp_px2.shape
    assert cnum == 2

    idxnext_p = (np.arange(pnum - 1, dtype=np.int32) + 1) % pnum
    pgtnext_px2 = pgtnp_px2[idxnext_p]
    pgtnp_px2 = pgtnp_px2[:-1]
    pnum = pnum - 1
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
    edgeidxsort_p = np.argsort(edgelen_p)

    edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
    for i in range(pnum):
        if edgenum[i] == 0:
            edgenum[i] = 1

    # after round, it may has 1 or 2 mismatch
    edgenumsum = np.sum(edgenum)
    if edgenumsum != newpnum:

        if edgenumsum > newpnum:

            id = -1
            passnum = edgenumsum - newpnum
            while passnum > 0:
                edgeid = edgeidxsort_p[id]
                if edgenum[edgeid] > passnum:
                    edgenum[edgeid] -= passnum
                    passnum -= passnum
                else:
                    passnum -= edgenum[edgeid] - 1
                    edgenum[edgeid] -= edgenum[edgeid] - 1
                    id -= 1
        else:
            id = -1
            edgeid = edgeidxsort_p[id]
            edgenum[edgeid] += newpnum - edgenumsum

    assert np.sum(edgenum) == newpnum

    psample = []
    for i in range(pnum):
        pb_1x2 = pgtnp_px2[i:i + 1]
        pe_1x2 = pgtnext_px2[i:i + 1]

        pnewnum = edgenum[i]
        wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i]

        pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
        psample.append(pmids)

    psamplenp = np.concatenate(psample, axis=0)
    return psamplenp


def img_poly_to_can_poly(img_poly, x_min, y_min, x_max, y_max):
    x_min, y_min = np.min(img_poly, axis=0)
    can_poly = img_poly - np.array([x_min, y_min])
    # h, w = y_max - y_min + 1, x_max - x_min + 1
    # long_side = max(h, w)
    # can_poly = can_poly / long_side
    return can_poly


def add_gaussian_noise(poly, x_min, y_min, x_max, y_max):
    h, w = y_max - y_min, x_max - x_min
    radius = data_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
    radius = max(0, int(radius))
    noise = np.random.uniform(-radius, radius, poly.shape)
    return poly + noise


def clip_poly_to_image(poly, h, w):
    poly[:, 0] = np.clip(poly[:, 0], a_min=0, a_max=w-1)
    poly[:, 1] = np.clip(poly[:, 1], a_min=0, a_max=h-1)
    return poly

