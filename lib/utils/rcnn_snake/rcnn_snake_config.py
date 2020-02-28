from lib.utils.snake.snake_config import *
from lib.config import cfg


cp_h, cp_w = 14, 56
roi_h, roi_w = 7, 28

nms_ct = True
max_ct_overlap = 0.7
ct_score = cfg.ct_score

cp_hm_nms = False
max_cp_det = 50
max_cp_overlap = 0.1
cp_score = 0.25

segm_or_bbox = 'segm'

