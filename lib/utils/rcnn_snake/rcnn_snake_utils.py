import torch
from lib.utils.snake.snake_decode import nms, topk, transpose_and_gather_feat
from lib.utils.rcnn_snake import rcnn_snake_config
from lib.csrc.extreme_utils import _ext


def box_to_roi(box, box_01):
    """ box: [b, n, 4] """
    box = box[box_01]
    ind = torch.cat([torch.full([box_01[i].sum()], i) for i in range(len(box_01))], dim=0)
    ind = ind.to(box.device).float()
    roi = torch.cat([ind[:, None], box], dim=1)
    return roi


def decode_cp_detection(cp_hm, cp_wh, abox, adet):
    batch, cat, height, width = cp_hm.size()
    if rcnn_snake_config.cp_hm_nms:
        cp_hm = nms(cp_hm)

    abox_w, abox_h = abox[..., 2] - abox[..., 0], abox[..., 3] - abox[..., 1]

    scores, inds, clses, ys, xs = topk(cp_hm, rcnn_snake_config.max_cp_det)
    cp_wh = transpose_and_gather_feat(cp_wh, inds)
    cp_wh = cp_wh.view(batch, rcnn_snake_config.max_cp_det, 2)

    cp_hm_h, cp_hm_w = cp_hm.size(2), cp_hm.size(3)

    xs = xs / cp_hm_w * abox_w[..., None] + abox[:, 0:1]
    ys = ys / cp_hm_h * abox_h[..., None] + abox[:, 1:2]
    boxes = torch.stack([xs - cp_wh[..., 0] / 2,
                         ys - cp_wh[..., 1] / 2,
                         xs + cp_wh[..., 0] / 2,
                         ys + cp_wh[..., 1] / 2], dim=2)

    ascore = adet[..., 4]
    acls = adet[..., 5]
    excluded_clses = [1, 2]
    for cls_ in excluded_clses:
        boxes[acls == cls_, 0] = abox[acls == cls_]
        scores[acls == cls_, 0] = 1
        scores[acls == cls_, 1:] = 0

    ct_num = len(abox)
    boxes_ = []
    for i in range(ct_num):
        cp_ind = _ext.nms(boxes[i], scores[i], rcnn_snake_config.max_cp_overlap)
        cp_01 = scores[i][cp_ind] > rcnn_snake_config.cp_score
        boxes_.append(boxes[i][cp_ind][cp_01])

    cp_ind = torch.cat([torch.full([len(boxes_[i])], i) for i in range(len(boxes_))], dim=0)
    cp_ind = cp_ind.to(boxes.device)
    boxes = torch.cat(boxes_, dim=0)

    return boxes, cp_ind

