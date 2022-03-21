import torch

def scheduler_LR(optimizer, milestone, gamma):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestone, gamma)
    return scheduler