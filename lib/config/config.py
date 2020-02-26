from .yacs import CfgNode as CN
import argparse
import os

cfg = CN()

# model
cfg.model = 'hello'
cfg.model_dir = 'data/model'

# network
cfg.network = 'dla_34'

# network heads
cfg.heads = ''

# task
cfg.task = ''

# gpus
cfg.gpus = [0]

# if load the pretrained network
cfg.resume = True


# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.dataset = 'CocoTrain'
cfg.train.epoch = 140
cfg.train.num_workers = 8

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-4
cfg.train.weight_decay = 5e-4

cfg.train.warmup = False
cfg.train.milestones = [80, 120, 200, 240]
cfg.train.gamma = 0.5

cfg.train.batch_size = 4

# test
cfg.test = CN()
cfg.test.dataset = 'CocoVal'
cfg.test.batch_size = 1
cfg.test.epoch = -1

# recorder
cfg.record_dir = 'data/record'

# result
cfg.result_dir = 'data/result'

# evaluation
cfg.skip_eval = False

cfg.save_ep = 5
cfg.eval_ep = 5


def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # assign the gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])

    if cfg.task == 'ct':
        num_classes = 8
        if 'Kins' in cfg.train.dataset:
            num_classes = 7
        cfg.heads = CN({'ct_hm': num_classes, 'wh': 2})

    if cfg.task == 'ext':
        num_classes = 8
        if 'Kins' in cfg.train.dataset:
            num_classes = 7
        cfg.heads = CN({'ct_hm': num_classes, 'ext': 8})

    if 'snake' in cfg.task:
        num_classes = 8
        if 'Sbd' in cfg.train.dataset:
            num_classes = 20
        if 'Linemod' in cfg.train.dataset:
            num_classes = 1
        if 'Kins' in cfg.train.dataset:
            num_classes = 7
        if 'Coco' in cfg.train.dataset:
            num_classes = 80
        cfg.heads = CN({'ct_hm': num_classes, 'wh': 2})
        # cfg.heads = CN({'ct_hm': num_classes, 'wh': 2, 'reg': 2})

        if cfg.task == 'ext_snake':
            cfg.heads = CN({'ct_hm': num_classes, 'ext': 8})

        if cfg.task == 'snake_only':
            cfg.heads = CN({})

    if cfg.task == 'dsnake':
        num_classes = 8
        cfg.heads = CN({'act_hm': num_classes, 'awh': 2, 'ct_hm': num_classes, 'wh': 2})

    if cfg.task == 'rcnn_snake':
        num_classes = 8
        cfg.heads = CN({'act_hm': num_classes, 'awh': 2})

    if cfg.task == 'retina':
        cfg.num_classes = 7

    cfg.det_dir = os.path.join(cfg.model_dir, cfg.task, args.det)

    # assign the network head conv
    cfg.head_conv = 64 if 'res' in cfg.network else 256

    cfg.model_dir = os.path.join(cfg.model_dir, cfg.task, cfg.model)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.model)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.model)


def make_cfg(args):
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    parse_cfg(cfg, args)
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
if len(args.type) > 0:
    cfg.task = "run"
cfg = make_cfg(args)
