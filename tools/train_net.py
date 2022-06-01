import argparse
import os
import sys
from os import mkdir

import torch.nn.functional as F

sys.path.append('../')
print(os.getcwd())
from config import cfg
from data import make_data_loader, get_val_data
from engine.example_trainer import do_train
from modeling import build_model
from solver import make_optimizer, make_scheduler

from loss import circle_loss, magface

from utils.logger import setup_logger

def train(cfg):
    model = build_model(cfg)
    device = cfg.device

    optimizer = make_optimizer(cfg, model)
    # scheduler = None
    scheduler = make_scheduler(cfg, optimizer)

    arguments = {}

    train_loader, num_class = make_data_loader(cfg, is_train=True)
    test_loader, test_num_class = make_data_loader(cfg, is_train=False)

    # get the validation data
    val_dataset = []
    val_labels = []
    for val_name in cfg.val_dataset:
        val_data, val_label = get_val_data(cfg.val_dataset_dir, val_name)
        val_dataset.append(val_data)
        val_labels.append(val_label)  

    # loss = circle_loss()
    loss = magface(cfg, num_class=num_class)

    do_train(
        cfg, 
        model,
        train_loader,
        val_dataset,
        val_labels,
        test_loader,
        optimizer,
        #None,
        scheduler,
        #F.cross_entropy
        loss
    )

def main():
    parser = argparse.ArgumentParser(description="Pytorch Template MNIST Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WOLRD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.model_dir
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)
    
    logger = setup_logger("template_model", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
        logger.info("Running with config:\n{}".format(cfg))
    
    train(cfg)

if __name__ == '__main__':
    main()
