import torch
from .Ir_scheduler import scheduler_LR

def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer

def make_scheduler(cfg, optimizer):
    gamma = cfg.SOLVER.GAMMA
    milestone = cfg.SOLVER.MILESTONES

    scheduler = scheduler_LR(optimizer, milestone, gamma)

    return scheduler
