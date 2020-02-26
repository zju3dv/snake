import torch
import numpy as np
from lib.utils.snake import snake_decode, snake_config
from lib.csrc.extreme_utils import _ext as extreme_utils
from lib.utils import data_utils


def collect_training(poly, ct_01):
    batch_size = ct_01.size(0)
    poly = torch.cat([poly[i][ct_01[i]] for i in range(batch_size)], dim=0)
    return poly


def prepare_training_init(ret, batch):
    ct_01 = batch['ct_01'].byte()
    init = {}
    init.update({'i_it_4py': collect_training(batch['i_it_4py'], ct_01)})
    init.update({'c_it_4py': collect_training(batch['c_it_4py'], ct_01)})
    init.update({'i_gt_4py': collect_training(batch['i_gt_4py'], ct_01)})
    init.update({'c_gt_4py': collect_training(batch['c_gt_4py'], ct_01)})

    ct_num = batch['meta']['ct_num']
    init.update({'ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})

    return init


def prepare_testing_init(box, score):
    i_it_4pys = snake_decode.get_init(box)
    i_it_4pys = uniform_upsample(i_it_4pys, snake_config.init_poly_num)
    c_it_4pys = img_poly_to_can_poly(i_it_4pys)

    ind = score > snake_config.ct_score
    i_it_4pys = i_it_4pys[ind]
    c_it_4pys = c_it_4pys[ind]
    ind = torch.cat([torch.full([ind[i].sum()], i) for i in range(ind.size(0))], dim=0)
    init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'ind': ind}

    return init


def get_box_match_ind(pred_box, score, gt_poly):
    if gt_poly.size(0) == 0:
        return [], []

    gt_box = torch.cat([torch.min(gt_poly, dim=1)[0], torch.max(gt_poly, dim=1)[0]], dim=1)
    iou_matrix = data_utils.box_iou(pred_box, gt_box)
    iou, gt_ind = iou_matrix.max(dim=1)
    box_ind = ((iou > snake_config.box_iou) * (score > snake_config.confidence)).nonzero().view(-1)
    gt_ind = gt_ind[box_ind]

    ind = np.unique(gt_ind.detach().cpu().numpy(), return_index=True)[1]
    box_ind = box_ind[ind]
    gt_ind = gt_ind[ind]

    return box_ind, gt_ind


def prepare_training_box(ret, batch, init):
    box = ret['detection'][..., :4]
    score = ret['detection'][..., 4]
    batch_size = box.size(0)
    i_gt_4py = batch['i_gt_4py']
    ct_01 = batch['ct_01'].byte()
    ind = [get_box_match_ind(box[i], score[i], i_gt_4py[i][ct_01[i]]) for i in range(batch_size)]
    box_ind = [ind_[0] for ind_ in ind]
    gt_ind = [ind_[1] for ind_ in ind]

    i_it_4py = torch.cat([snake_decode.get_init(box[i][box_ind[i]][None]) for i in range(batch_size)], dim=1)
    if i_it_4py.size(1) == 0:
        return

    i_it_4py = uniform_upsample(i_it_4py, snake_config.init_poly_num)[0]
    c_it_4py = img_poly_to_can_poly(i_it_4py)
    i_gt_4py = torch.cat([batch['i_gt_4py'][i][gt_ind[i]] for i in range(batch_size)], dim=0)
    c_gt_4py = torch.cat([batch['c_gt_4py'][i][gt_ind[i]] for i in range(batch_size)], dim=0)
    init_4py = {'i_it_4py': i_it_4py, 'c_it_4py': c_it_4py, 'i_gt_4py': i_gt_4py, 'c_gt_4py': c_gt_4py}

    i_it_py = snake_decode.get_octagon(i_gt_4py[None])
    i_it_py = uniform_upsample(i_it_py, snake_config.poly_num)[0]
    c_it_py = img_poly_to_can_poly(i_it_py)
    i_gt_py = torch.cat([batch['i_gt_py'][i][gt_ind[i]] for i in range(batch_size)], dim=0)
    init_py = {'i_it_py': i_it_py, 'c_it_py': c_it_py, 'i_gt_py': i_gt_py}

    ind = torch.cat([torch.full([len(gt_ind[i])], i) for i in range(batch_size)], dim=0)

    if snake_config.train_pred_box_only:
        for k, v in init_4py.items():
            init[k] = v
        for k, v in init_py.items():
            init[k] = v
        init['4py_ind'] = ind
        init['py_ind'] = ind
    else:
        init.update({k: torch.cat([init[k], v], dim=0) for k, v in init_4py.items()})
        init.update({'4py_ind': torch.cat([init['4py_ind'], ind], dim=0)})
        init.update({k: torch.cat([init[k], v], dim=0) for k, v in init_py.items()})
        init.update({'py_ind': torch.cat([init['py_ind'], ind], dim=0)})


def prepare_training(ret, batch):
    ct_01 = batch['ct_01'].byte()
    init = {}
    init.update({'i_it_4py': collect_training(batch['i_it_4py'], ct_01)})
    init.update({'c_it_4py': collect_training(batch['c_it_4py'], ct_01)})
    init.update({'i_gt_4py': collect_training(batch['i_gt_4py'], ct_01)})
    init.update({'c_gt_4py': collect_training(batch['c_gt_4py'], ct_01)})

    init.update({'i_it_py': collect_training(batch['i_it_py'], ct_01)})
    init.update({'c_it_py': collect_training(batch['c_it_py'], ct_01)})
    init.update({'i_gt_py': collect_training(batch['i_gt_py'], ct_01)})
    init.update({'c_gt_py': collect_training(batch['c_gt_py'], ct_01)})

    ct_num = batch['meta']['ct_num']
    init.update({'4py_ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})
    init.update({'py_ind': init['4py_ind']})

    if snake_config.train_pred_box:
        prepare_training_box(ret, batch, init)

    init['4py_ind'] = init['4py_ind'].to(ct_01.device)
    init['py_ind'] = init['py_ind'].to(ct_01.device)

    return init


def prepare_training_evolve(ex, init):
    if not snake_config.train_pred_ex:
        evolve = {'i_it_py': init['i_it_py'], 'c_it_py': init['c_it_py'], 'i_gt_py': init['i_gt_py']}
        return evolve

    i_gt_py = init['i_gt_py']

    if snake_config.train_nearest_gt:
        shift = -(ex[:, :1] - i_gt_py).pow(2).sum(2).argmin(1)
        i_gt_py = extreme_utils.roll_array(i_gt_py, shift)

    i_it_py = snake_decode.get_octagon(ex[None])
    i_it_py = uniform_upsample(i_it_py, snake_config.poly_num)[0]
    c_it_py = img_poly_to_can_poly(i_it_py)
    evolve = {'i_it_py': i_it_py, 'c_it_py': c_it_py, 'i_gt_py': i_gt_py}

    return evolve


def prepare_testing_evolve(ex):
    if len(ex) == 0:
        i_it_pys = torch.zeros([0, snake_config.poly_num, 2]).to(ex)
        c_it_pys = torch.zeros_like(i_it_pys)
    else:
        i_it_pys = snake_decode.get_octagon(ex[None])
        i_it_pys = uniform_upsample(i_it_pys, snake_config.poly_num)[0]
        c_it_pys = img_poly_to_can_poly(i_it_pys)
    evolve = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys}
    return evolve


def get_gcn_feature(cnn_feature, img_poly, ind, h, w):
    img_poly = img_poly.clone()
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1

    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device)
    for i in range(batch_size):
        poly = img_poly[ind == i].unsqueeze(0)
        feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], poly)[0].permute(1, 0, 2)
        gcn_feature[ind == i] = feature

    return gcn_feature


def get_adj_mat(n_adj, n_nodes, device):
    a = np.zeros([n_nodes, n_nodes])

    for i in range(n_nodes):
        for j in range(-n_adj // 2, n_adj // 2 + 1):
            if j != 0:
                a[i][(i + j) % n_nodes] = 1
                a[(i + j) % n_nodes][i] = 1

    a = torch.Tensor(a.astype(np.float32))
    return a.to(device)


def get_adj_ind(n_adj, n_nodes, device):
    ind = torch.LongTensor([i for i in range(-n_adj // 2, n_adj // 2 + 1) if i != 0])
    ind = (torch.arange(n_nodes)[:, None] + ind[None]) % n_nodes
    return ind.to(device)


def get_pconv_ind(n_adj, n_nodes, device):
    n_outer_nodes = snake_config.poly_num
    ind = torch.LongTensor([i for i in range(-n_adj // 2, n_adj // 2 + 1)])
    outer_ind = (torch.arange(n_outer_nodes)[:, None] + ind[None]) % n_outer_nodes
    inner_ind = outer_ind + n_outer_nodes
    ind = torch.cat([outer_ind, inner_ind], dim=1)
    return ind


def img_poly_to_can_poly(img_poly):
    if len(img_poly) == 0:
        return torch.zeros_like(img_poly)
    x_min = torch.min(img_poly[..., 0], dim=-1)[0]
    y_min = torch.min(img_poly[..., 1], dim=-1)[0]
    can_poly = img_poly.clone()
    can_poly[..., 0] = can_poly[..., 0] - x_min[..., None]
    can_poly[..., 1] = can_poly[..., 1] - y_min[..., None]
    # x_max = torch.max(img_poly[..., 0], dim=-1)[0]
    # y_max = torch.max(img_poly[..., 1], dim=-1)[0]
    # h, w = y_max - y_min + 1, x_max - x_min + 1
    # long_side = torch.max(h, w)
    # can_poly = can_poly / long_side[..., None, None]
    return can_poly


def uniform_upsample(poly, p_num):
    # 1. assign point number for each edge
    # 2. calculate the coefficient for linear interpolation
    next_poly = torch.roll(poly, -1, 2)
    edge_len = (next_poly - poly).pow(2).sum(3).sqrt()
    edge_num = torch.round(edge_len * p_num / torch.sum(edge_len, dim=2)[..., None]).long()
    edge_num = torch.clamp(edge_num, min=1)
    edge_num_sum = torch.sum(edge_num, dim=2)
    edge_idx_sort = torch.argsort(edge_num, dim=2, descending=True)
    extreme_utils.calculate_edge_num(edge_num, edge_num_sum, edge_idx_sort, p_num)
    edge_num_sum = torch.sum(edge_num, dim=2)
    assert torch.all(edge_num_sum == p_num)

    edge_start_idx = torch.cumsum(edge_num, dim=2) - edge_num
    weight, ind = extreme_utils.calculate_wnp(edge_num, edge_start_idx, p_num)
    poly1 = poly.gather(2, ind[..., 0:1].expand(ind.size(0), ind.size(1), ind.size(2), 2))
    poly2 = poly.gather(2, ind[..., 1:2].expand(ind.size(0), ind.size(1), ind.size(2), 2))
    poly = poly1 * (1 - weight) + poly2 * weight

    return poly


def zoom_poly(poly, scale):
    mean = (poly.min(dim=1, keepdim=True)[0] + poly.max(dim=1, keepdim=True)[0]) * 0.5
    poly = poly - mean
    poly = poly * scale + mean
    return poly

