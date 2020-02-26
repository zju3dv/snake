import numpy as np
import cv2
import pycocotools.mask as mask_utils
from PIL import Image
import os


def poly_to_mask(poly, label, h, w):
    mask = []
    for i in range(len(poly)):
        mask_ = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_, [np.round(poly[i]).astype(int)], int(label[i]))
        mask.append(mask_)
    return mask


def coco_poly_to_mask(poly, h, w):
    mask = []
    for i in range(len(poly)):
        rles = mask_utils.frPyObjects([poly[i].reshape(-1)], h, w)
        rle = mask_utils.merge(rles)
        mask_ = mask_utils.decode(rle)
        mask.append(mask_)
    return mask


def rcnn_poly_to_mask(poly, ind_group, label, h, w):
    mask = []
    for i in range(len(ind_group)):
        poly_ = [np.round(poly[ind]).astype(int) for ind in ind_group[i]]
        mask_ = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_, poly_, int(label[i]))
        mask.append(mask_)
    return mask


def rcnn_coco_poly_to_mask(poly, ind_group, h, w):
    mask = []
    for i in range(len(ind_group)):
        poly_ = [poly[ind].reshape(-1) for ind in ind_group[i]]
        rles = mask_utils.frPyObjects(poly_, h, w)
        rle = mask_utils.merge(rles)
        mask_ = mask_utils.decode(rle)
        mask.append(mask_)
    return mask


def coco_poly_to_rle(poly, h, w):
    rle_ = []
    for i in range(len(poly)):
        rles = mask_utils.frPyObjects([poly[i].reshape(-1)], h, w)
        rle = mask_utils.merge(rles)
        rle['counts'] = rle['counts'].decode('utf-8')
        rle_.append(rle)
    return rle_


def rcnn_coco_poly_to_rle(poly, ind_group, h, w):
    rle_ = []
    for i in range(len(ind_group)):
        poly_ = [poly[ind].reshape(-1) for ind in ind_group[i]]
        rles = mask_utils.frPyObjects(poly_, h, w)
        rle = mask_utils.merge(rles)
        rle['counts'] = rle['counts'].decode('utf-8')
        rle_.append(rle)
    return rle_

