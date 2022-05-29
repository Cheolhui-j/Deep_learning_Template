import torch
from .lr_scheduler import scheduler_LR

import sys

sys.path.append('../')
from config import cfg
from modeling import build_model

def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.init_lr
        weight_decay = cfg.wd
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = getattr(torch.optim, cfg.optimizer)(params, momentum=cfg.mom)
    return optimizer

def make_scheduler(cfg, optimizer):
    gamma = cfg.lr_decay_ratio
    milestone = cfg.lr_decay_epoch

    scheduler = scheduler_LR(optimizer, milestone, gamma)

    return scheduler

if __name__ == "__main__":

    model = build_model(cfg) 

    opt = make_optimizer(cfg, model)

    scheduler = make_scheduler(cfg, opt)

    print('optimizer : ', opt)
    print('schedular : ', scheduler)