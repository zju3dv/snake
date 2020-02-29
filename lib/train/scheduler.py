from torch.optim.lr_scheduler import MultiStepLR
from collections import Counter
from lib.utils.optimizer.lr_scheduler import WarmupMultiStepLR, ManualStepLR


def make_lr_scheduler(cfg, optimizer):
    if cfg.train.warmup:
        scheduler = WarmupMultiStepLR(optimizer, cfg.train.milestones, cfg.train.gamma, 1.0/3, 5, 'linear')
    elif cfg.train.scheduler == 'manual':
        scheduler = ManualStepLR(optimizer, milestones=cfg.train.milestones, gammas=cfg.train.gammas)
    else:
        scheduler = MultiStepLR(optimizer, milestones=cfg.train.milestones, gamma=cfg.train.gamma)
    return scheduler


def set_lr_scheduler(cfg, scheduler):
    if cfg.train.warmup:
        scheduler.milestones = cfg.train.milestones
    else:
        scheduler.milestones = Counter(cfg.train.milestones)
    scheduler.gamma = cfg.train.gamma

