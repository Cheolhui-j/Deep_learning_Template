   
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

config = CN()

# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

config.network = 'MobileResNet100v2'
config.pretrained = ''

# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------

config.loss = 'MagFace'
config.optimizer = 'SGD'

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

config.train_dataset = 'ms1m_arcface'
config.train_dataset_type = 'lmdb'
config.val_dataset = ['lfw', 'agedb_30', 'cfp_fp', 'cfp_ff', 'calfw', 'cplfw']

# -----------------------------------------------------------------------------
# directory
# -----------------------------------------------------------------------------

config.train_dataset_dir = '/workspace/hdd2/datasets_fr/train/ms1m_arcface'
config.val_dataset_dir = '/workspace/hdd2/FR/datasets/validation'
config.test_dataset_dir = '/workspace/hdd2/FR/datasets/test'
config.model_dir = './models'

# ---------------------------------------------------------------------------- #
# Hyper params
# ---------------------------------------------------------------------------- #

config.batch_size = 32
config.init_lr = 0.1
config.lr_scheduler = 'MultiStep'
config.lr_decay_epoch = [8, 16, 24, 32, 40]
config.lr_decay_ratio = 0.1
config.num_workers = 8
config.num_epoch = 20
config.device = 'cuda'
config.view_freq = 20
config.view_valid_freq = 1

# ---------------------------------------------------------------------------- #
# Network definition
# ---------------------------------------------------------------------------- #

network = CN()

# --------------------------------------------
# Mobile Residual Network (MobileResNet) configurations
# --------------------------------------------
network.MobileResNet34 = CN()
network.MobileResNet34.network_name = 'MobileResNet34'
network.MobileResNet34.input_opt = 'Stem-S' # option: 'O' or 'L'
network.MobileResNet34.output_opt = 'E' # option: 'O' or 'E'
network.MobileResNet34.block_opt = 'IR'

network.MobileResNet50 = CN()
network.MobileResNet50.network_name = 'MobileResNet50'
network.MobileResNet50.input_opt = 'Stem-S' # option: 'O' or 'L'
network.MobileResNet50.output_opt = 'E' # option: 'O' or 'E'
network.MobileResNet50.block_opt = 'IR'

# --------------------------------------------
# Residual Network (ResNet) configurations
# --------------------------------------------
network.ResNet18 = CN()
network.ResNet18.network_name = 'ResNet18'
network.ResNet18.input_opt = 'L' # option: 'O' or 'L' # what is 'DS' ?
network.ResNet18.output_opt = 'E' # option: 'O' or 'E'
network.ResNet18.block_opt = 'IR' # option: 'O' or 'IR'
network.ResNet18.use_se = False

# --------------------------------------------
# Mobile Residual Network version2 (MobileResNetv2) configurations
# --------------------------------------------
network.MobileResNet34v2 = CN()
network.MobileResNet34v2.network_name = 'MobileResNet34v2'
network.MobileResNet34v2.input_opt = 'Stem-S' # option: 'O' or 'L'
network.MobileResNet34v2.output_opt = 'E' # option: 'O' or 'E'
network.MobileResNet34v2.block_opt = 'IR'
network.MobileResNet34v2.width = 1.0
network.MobileResNet34v2.expansion = 6

network.MobileResNet50v2 = CN()
network.MobileResNet50v2.network_name = 'MobileResNet50v2'
network.MobileResNet50v2.input_opt = 'Stem-S' # option: 'O' or 'L'
network.MobileResNet50v2.output_opt = 'E' # option: 'O' or 'E'
network.MobileResNet50v2.block_opt = 'IR'
network.MobileResNet50v2.width = 1.0
network.MobileResNet50v2.expansion = 6

network.MobileResNet100v2 = CN()
network.MobileResNet100v2.network_name = 'MobileResNet100v2'
network.MobileResNet100v2.input_opt = 'Stem-L' # option: 'O' or 'L'
network.MobileResNet100v2.output_opt = 'E' # option: 'O' or 'E'
network.MobileResNet100v2.block_opt = 'IR'
network.MobileResNet100v2.width = [1.2, 1.2, 1.2, 1.2]
network.MobileResNet100v2.expansion = 6

network.MobileResNet100v3 = CN()
network.MobileResNet100v3.network_name = 'MobileResNet100v3'
network.MobileResNet100v3.input_opt = 'Stem-L' # option: 'O' or 'L'
network.MobileResNet100v3.output_opt = 'E' # option: 'O' or 'E'
network.MobileResNet100v3.block_opt = 'IR'
network.MobileResNet100v3.width = [1.2, 1.2, 1.2, 1.2]
network.MobileResNet100v3.expansion = 6

network.MobileResNet160v3 = CN()
network.MobileResNet160v3.network_name = 'MobileResNet160v3'
network.MobileResNet160v3.input_opt = 'Stem-L' # option: 'O' or 'L'
network.MobileResNet160v3.output_opt = 'E' # option: 'O' or 'E'
network.MobileResNet160v3.block_opt = 'IR'
network.MobileResNet160v3.width = [1.2, 1.2, 1.3, 1.3]
network.MobileResNet160v3.expansion = 6

# ---------------------------------------------------------------------------- #
# Loss definition
# ---------------------------------------------------------------------------- #

loss = CN()

# --------------------------------------------
# MagFace loss configurations
# --------------------------------------------
loss.MagFace = CN()
loss.MagFace.loss_name = 'MagFace'
loss.MagFace.emd_size = 512
loss.MagFace.loss_s = 64.0
loss.MagFace.u_a = 110
loss.MagFace.l_a = 10
loss.MagFace.u_m = 0.8
loss.MagFace.l_m = 0.4
loss.MagFace.easy_margin = False

# --------------------------------------------
# Circle loss configurations
# --------------------------------------------
loss.Circle_loss = CN()
loss.Circle_loss.loss_name = 'Circle_loss'
## circle not complete ! ##


# ==================================== Optimizer configuration ====================================
optimizer = CN()

# --------------------------------------------
# SGD configurations
# --------------------------------------------
optimizer.SGD = CN()
optimizer.SGD.optimizer_name = 'SGD'
optimizer.SGD.wd = 0.0005
optimizer.SGD.mom = 0.9
# --------------------------------------------
# Adam configurations
# --------------------------------------------
optimizer.Adam = CN()
optimizer.Adam.optimizer_name = 'Adam'
optimizer.Adam.wd = 0.0005

def generate_config(_network, _loss, _optimizer):
    for k, v in loss[_loss].items():
        config[k] = v
    for k, v in optimizer[_optimizer].items():
        config[k] = v
    for k, v in network[_network].items():
        config[k] = v
    config.loss = _loss
    config.optimizer = _optimizer
    config.network = _network

generate_config(config.network, config.loss, config.optimizer)

print(config)