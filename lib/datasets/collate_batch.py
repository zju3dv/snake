from torch.utils.data.dataloader import default_collate
import torch
import numpy as np


def snake_collator(batch):
    ret = {'inp': default_collate([b['inp'] for b in batch])}

    meta = default_collate([b['meta'] for b in batch])
    ret.update({'meta': meta})

    if 'test' in meta:
        return ret

    # detection
    ct_hm = default_collate([b['ct_hm'] for b in batch])

    batch_size = len(batch)
    ct_num = torch.max(meta['ct_num'])
    wh = torch.zeros([batch_size, ct_num, 2], dtype=torch.float)
    # reg = torch.zeros([batch_size, ct_num, 2], dtype=torch.float)
    ct_cls = torch.zeros([batch_size, ct_num], dtype=torch.int64)
    ct_ind = torch.zeros([batch_size, ct_num], dtype=torch.int64)
    ct_01 = torch.zeros([batch_size, ct_num], dtype=torch.uint8)
    for i in range(batch_size):
        ct_01[i, :meta['ct_num'][i]] = 1

    if ct_num != 0:
        wh[ct_01] = torch.Tensor(sum([b['wh'] for b in batch], []))
        # reg[ct_01] = torch.Tensor(sum([b['reg'] for b in batch], []))
        ct_cls[ct_01] = torch.LongTensor(sum([b['ct_cls'] for b in batch], []))
        ct_ind[ct_01] = torch.LongTensor(sum([b['ct_ind'] for b in batch], []))

    detection = {'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind, 'ct_01': ct_01.float()}
    # detection = {'ct_hm': ct_hm, 'wh': wh, 'reg': reg, 'ct_cls': ct_cls, 'ct_ind': ct_ind, 'ct_01': ct_01.float()}
    ret.update(detection)

    from lib.utils.snake import snake_config

    # init
    i_it_4pys = torch.zeros([batch_size, ct_num, snake_config.init_poly_num, 2], dtype=torch.float)
    c_it_4pys = torch.zeros([batch_size, ct_num, snake_config.init_poly_num, 2], dtype=torch.float)
    i_gt_4pys = torch.zeros([batch_size, ct_num, 4, 2], dtype=torch.float)
    c_gt_4pys = torch.zeros([batch_size, ct_num, 4, 2], dtype=torch.float)
    if ct_num != 0:
        i_it_4pys[ct_01] = torch.Tensor(sum([b['i_it_4py'] for b in batch], []))
        c_it_4pys[ct_01] = torch.Tensor(sum([b['c_it_4py'] for b in batch], []))
        i_gt_4pys[ct_01] = torch.Tensor(sum([b['i_gt_4py'] for b in batch], []))
        c_gt_4pys[ct_01] = torch.Tensor(sum([b['c_gt_4py'] for b in batch], []))
    init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
    ret.update(init)

    # evolution
    i_it_pys = torch.zeros([batch_size, ct_num, snake_config.poly_num, 2], dtype=torch.float)
    c_it_pys = torch.zeros([batch_size, ct_num, snake_config.poly_num, 2], dtype=torch.float)
    i_gt_pys = torch.zeros([batch_size, ct_num, snake_config.gt_poly_num, 2], dtype=torch.float)
    c_gt_pys = torch.zeros([batch_size, ct_num, snake_config.gt_poly_num, 2], dtype=torch.float)
    if ct_num != 0:
        i_it_pys[ct_01] = torch.Tensor(sum([b['i_it_py'] for b in batch], []))
        c_it_pys[ct_01] = torch.Tensor(sum([b['c_it_py'] for b in batch], []))
        i_gt_pys[ct_01] = torch.Tensor(sum([b['i_gt_py'] for b in batch], []))
        c_gt_pys[ct_01] = torch.Tensor(sum([b['c_gt_py'] for b in batch], []))
    evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}
    ret.update(evolution)

    return ret


def dsnake_collator(batch):
    ret = snake_collator(batch)
    meta = ret['meta']

    # detection
    act_hm = default_collate([b['act_hm'] for b in batch])

    batch_size = len(batch)
    act_num = torch.max(meta['act_num'])
    awh = torch.zeros([batch_size, act_num, 2], dtype=torch.float)
    act_ind = torch.zeros([batch_size, act_num], dtype=torch.int64)
    act_01 = torch.zeros([batch_size, act_num], dtype=torch.uint8)
    for i in range(batch_size):
        act_01[i, :meta['act_num'][i]] = 1

    awh[act_01] = torch.Tensor(sum([b['awh'] for b in batch], []))
    act_ind[act_01] = torch.LongTensor(sum([b['act_ind'] for b in batch], []))

    adet = {'act_hm': act_hm, 'awh': awh, 'act_ind': act_ind, 'act_01': act_01.float()}
    ret.update(adet)

    return ret


def rcnn_snake_collator(batch):
    ret = {'inp': default_collate([b['inp'] for b in batch])}

    meta = default_collate([b['meta'] for b in batch])
    ret.update({'meta': meta})

    if 'test' in meta:
        return ret

    # detection
    act_hm = default_collate([b['act_hm'] for b in batch])

    batch_size = len(batch)
    act_num = torch.max(meta['act_num'])
    awh = torch.zeros([batch_size, act_num, 2], dtype=torch.float)
    act_ind = torch.zeros([batch_size, act_num], dtype=torch.int64)
    act_01 = torch.zeros([batch_size, act_num], dtype=torch.uint8)
    for i in range(batch_size):
        act_01[i, :meta['act_num'][i]] = 1

    if act_num != 0:
        awh[act_01] = torch.Tensor(sum([b['awh'] for b in batch], []))
        act_ind[act_01] = torch.LongTensor(sum([b['act_ind'] for b in batch], []))

    detection = {'act_hm': act_hm, 'awh': awh, 'act_ind': act_ind, 'act_01': act_01.float()}
    ret.update(detection)

    from lib.utils.rcnn_snake import rcnn_snake_config as snake_config

    ct_num = torch.max(meta['ct_num'])
    ct_01 = torch.zeros([batch_size, ct_num], dtype=torch.uint8)
    for i in range(batch_size):
        ct_01[i, :meta['ct_num'][i]] = 1
    ret.update({'ct_01': ct_01.float()})

    # component detection
    cp_hm = default_collate(sum([[hm for hm in b['cp_hm']] for b in batch], []))

    cp_num_list = sum([[len(wh) for wh in b['cp_wh']] for b in batch], [])
    cp_num = max(cp_num_list)
    cp_wh = torch.zeros([len(cp_hm), cp_num, 2], dtype=torch.float)
    cp_ind = torch.zeros([len(cp_hm), cp_num], dtype=torch.int64)
    cp_01 = torch.zeros([len(cp_hm), cp_num], dtype=torch.uint8)
    for i in range(len(cp_hm)):
        cp_01[i, :cp_num_list[i]] = 1

    if cp_num != 0:
        cp_wh[cp_01] = torch.Tensor(sum(sum([b['cp_wh'] for b in batch], []), []))
        cp_ind[cp_01] = torch.LongTensor(sum(sum([b['cp_ind'] for b in batch], []), []))

    cp_hm_ = torch.zeros([batch_size, act_num, 1, snake_config.cp_h, snake_config.cp_w], dtype=torch.float)
    cp_wh_ = torch.zeros([batch_size, act_num, cp_num, 2], dtype=torch.float)
    cp_ind_ = torch.zeros([batch_size, act_num, cp_num], dtype=torch.int64)
    cp_01_ = torch.zeros([batch_size, act_num, cp_num], dtype=torch.uint8)

    cp_hm_[act_01] = cp_hm
    cp_wh_[act_01] = cp_wh
    cp_ind_[act_01] = cp_ind
    cp_01_[act_01] = cp_01

    cp_detection = {'cp_hm': cp_hm_, 'cp_wh': cp_wh_, 'cp_ind': cp_ind_, 'cp_01': cp_01_.float()}
    ret.update(cp_detection)

    # init
    i_it_4pys = torch.zeros([batch_size, ct_num, snake_config.init_poly_num, 2], dtype=torch.float)
    c_it_4pys = torch.zeros([batch_size, ct_num, snake_config.init_poly_num, 2], dtype=torch.float)
    i_gt_4pys = torch.zeros([batch_size, ct_num, 4, 2], dtype=torch.float)
    c_gt_4pys = torch.zeros([batch_size, ct_num, 4, 2], dtype=torch.float)
    if ct_num != 0:
        i_it_4pys[ct_01] = torch.Tensor(sum([b['i_it_4py'] for b in batch], []))
        c_it_4pys[ct_01] = torch.Tensor(sum([b['c_it_4py'] for b in batch], []))
        i_gt_4pys[ct_01] = torch.Tensor(sum([b['i_gt_4py'] for b in batch], []))
        c_gt_4pys[ct_01] = torch.Tensor(sum([b['c_gt_4py'] for b in batch], []))
    init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
    ret.update(init)

    # evolution
    i_it_pys = torch.zeros([batch_size, ct_num, snake_config.poly_num, 2], dtype=torch.float)
    c_it_pys = torch.zeros([batch_size, ct_num, snake_config.poly_num, 2], dtype=torch.float)
    i_gt_pys = torch.zeros([batch_size, ct_num, snake_config.gt_poly_num, 2], dtype=torch.float)
    c_gt_pys = torch.zeros([batch_size, ct_num, snake_config.gt_poly_num, 2], dtype=torch.float)
    if ct_num != 0:
        i_it_pys[ct_01] = torch.Tensor(sum([b['i_it_py'] for b in batch], []))
        c_it_pys[ct_01] = torch.Tensor(sum([b['c_it_py'] for b in batch], []))
        i_gt_pys[ct_01] = torch.Tensor(sum([b['i_gt_py'] for b in batch], []))
        c_gt_pys[ct_01] = torch.Tensor(sum([b['c_gt_py'] for b in batch], []))
    evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}
    ret.update(evolution)

    return ret


def ext_snake_collator(batch):
    ret = {'inp': default_collate([b['inp'] for b in batch])}

    meta = default_collate([b['meta'] for b in batch])
    ret.update({'meta': meta})

    if 'test' in meta:
        return ret

    # detection
    ct_hm = default_collate([b['ct_hm'] for b in batch])

    batch_size = len(batch)
    ct_num = torch.max(meta['ct_num'])
    ext = torch.zeros([batch_size, ct_num, 8], dtype=torch.float)
    ct_cls = torch.zeros([batch_size, ct_num], dtype=torch.int64)
    ct_ind = torch.zeros([batch_size, ct_num], dtype=torch.int64)
    ct_01 = torch.zeros([batch_size, ct_num], dtype=torch.uint8)
    for i in range(batch_size):
        ct_01[i, :meta['ct_num'][i]] = 1

    ext[ct_01] = torch.Tensor(sum([b['ext'] for b in batch], []))
    ct_cls[ct_01] = torch.LongTensor(sum([b['ct_cls'] for b in batch], []))
    ct_ind[ct_01] = torch.LongTensor(sum([b['ct_ind'] for b in batch], []))

    detection = {'ct_hm': ct_hm, 'ext': ext, 'ct_cls': ct_cls, 'ct_ind': ct_ind, 'ct_01': ct_01.float()}
    ret.update(detection)

    from lib.utils.snake import snake_config

    # init
    i_it_4pys = torch.zeros([batch_size, ct_num, snake_config.init_poly_num, 2], dtype=torch.float)
    c_it_4pys = torch.zeros([batch_size, ct_num, snake_config.init_poly_num, 2], dtype=torch.float)
    i_gt_4pys = torch.zeros([batch_size, ct_num, 4, 2], dtype=torch.float)
    c_gt_4pys = torch.zeros([batch_size, ct_num, 4, 2], dtype=torch.float)
    i_it_4pys[ct_01] = torch.Tensor(sum([b['i_it_4py'] for b in batch], []))
    c_it_4pys[ct_01] = torch.Tensor(sum([b['c_it_4py'] for b in batch], []))
    i_gt_4pys[ct_01] = torch.Tensor(sum([b['i_gt_4py'] for b in batch], []))
    c_gt_4pys[ct_01] = torch.Tensor(sum([b['c_gt_4py'] for b in batch], []))
    init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
    ret.update(init)

    # evolution
    i_it_pys = torch.zeros([batch_size, ct_num, snake_config.poly_num, 2], dtype=torch.float)
    c_it_pys = torch.zeros([batch_size, ct_num, snake_config.poly_num, 2], dtype=torch.float)
    i_gt_pys = torch.zeros([batch_size, ct_num, snake_config.gt_poly_num, 2], dtype=torch.float)
    c_gt_pys = torch.zeros([batch_size, ct_num, snake_config.gt_poly_num, 2], dtype=torch.float)
    i_it_pys[ct_01] = torch.Tensor(sum([b['i_it_py'] for b in batch], []))
    c_it_pys[ct_01] = torch.Tensor(sum([b['c_it_py'] for b in batch], []))
    i_gt_pys[ct_01] = torch.Tensor(sum([b['i_gt_py'] for b in batch], []))
    c_gt_pys[ct_01] = torch.Tensor(sum([b['c_gt_py'] for b in batch], []))
    evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}
    ret.update(evolution)

    return ret


_collators = {
    'snake': snake_collator,
    'ct': snake_collator,
    'rcnn_snake': rcnn_snake_collator,
    'ct_rcnn': rcnn_snake_collator
}


def make_collator(cfg):
    if cfg.task in _collators:
        return _collators[cfg.task]
    else:
        return default_collate

