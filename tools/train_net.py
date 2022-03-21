import argparse
import os
import sys
from os import mkdir

import torch.nn.functional as F

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.example_trainer import do_train
from modeling import build_model
from solver import make_optimizer, make_scheduler

from loss import circle_loss

from utils.logger import setup_logger

# evaluation 값 다르니까 확인하고 아마 모델 형식도 좀 바꿔야 될 듯.
# 우선은 loss를 내 꺼말고 github 꺼로 돌려서 확인하는 것도 좋을 듯.

# 02.27 
# circloss 및 evaluator는 작동하는데 학습이 덜 되어서 그런지 
# precision값이 매우 안 좋게 나옴.
# 해서 
# 1. 그냥 작은 네트워크 renet으로 교체하고 확인.
# 2. 1번이 통과되면 좀 오래 학습을 진행 시켜본다. epoch 20에 step LR로
# 가 합리적일 거 같음. 우선은 1번부터 진행해보길...
# 만약 1번 결과가 안 좋다면 circle말고 cross_entropy로 진행해보길...

# 02.28
# 어제 명시한대로 진행했고 역시 학습이 덜 된 상태였음. 
# 일단 precision 82 , recall 87 까지 봤음.
# 이제 training 후에 accuracy 나오는 거 고치면 될 거 같음.
# 이건 차후에 ignite 좀 더 학습하고 나서 해보는 게 좋을 듯.
# tensorboard 적용하고 아마 load model은 없는 거 같은데 확인해보고 추가하는 게 좋을 듯
# 그리고 참고로 scheduler도 없어서 추가함. 이 과정에서 ignite 버전 0.4.8됨. 

# 03.01
# tensorboard loss에 대해서 추가함.
# 진행하다보니 아무래도 tensorboard를 하려면 미루었던 evaluate를 ignite 방식으로 재정의할 필요가 있어보임.
# 일단 acc, precision, recall 전부 해봤는데 나름 나오는 거 같긴 한데 약간 야메임.
# 이 부분 보고 정리할 필요가 있음.

# 03.02
# 주로 코드 정리 쪽을 했음.
# 이제 어느 정도 코드가 완성됐으니 여기까지 한 거 정리하고 직접 서버에서 돌려보고
# 잘 동작하면 얼굴 데이터셋에 대해서도 작동해보는 게 좋을 듯.
# 내일은 기록한 내용, 검색한 내용 및 코드 정리 마무리 바람. 

# 03.10
# roc 계산 코드랑 custom 데이터 로드 코드 추가.
# 두 개 다 작동하는지는 아직 잘 모르겠고 차후 서버에서 학습할 떄 확인예정.
# ROC 같은 경우에는 작동 확인까지는 되었는데 이게 맞는 값인지 확인이 필요한 상황.
# 그리고 config 파일 세분화? 조직화?가 좀더 필요해보임. 
# 일단은 이정도에서 git에 push 할까 생각 중.

def train(cfg):
    model = build_model(cfg)
    device = cfg.MODEL.DEVICE

    optimizer = make_optimizer(cfg, model)
    # scheduler = None
    scheduler = make_scheduler(cfg, optimizer)

    arguments = {}

    train_loader = make_data_loader(cfg, is_train=True)
    val_loader = make_data_loader(cfg, is_train=False)

    loss = circle_loss()

    do_train(
        cfg, 
        model,
        train_loader,
        val_loader,
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

    output_dir = cfg.OUTPUT_DIR
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
