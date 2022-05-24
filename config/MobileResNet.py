   
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# _N
# -----------------------------------------------------------------------------

_C.network = 'ResNet18'
_C.checkpoint = ''

# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------

_C.loss = 'magface'
_C.optimizer = 'SGD'

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

_C.train_dataset = 'ms1m_arcface'
_C.train_dataset_type = 'img'
_C.val_dataset = ['lfw', 'agedb_30', 'cfp_fp', 'cfp_ff', 'calfw', 'cplfw']

# -----------------------------------------------------------------------------
# directory
# -----------------------------------------------------------------------------

_C.train_dataset_dir = '/workspace/hdd2/FR/datasets/train/ms1m_arcface'
_C.val_dataset_dir = '/workspace/hdd2/FR/datasets/validation'
_C.test_dataset_dir = '/workspace/hdd2/FR/datasets/test'
_C.model_dir = './models'

# ---------------------------------------------------------------------------- #
# Hyper params
# ---------------------------------------------------------------------------- #

_C.batch_size = 512
_C.init_lr = 0.1
_C.lr_scheduler = 'MultiStep'
_C.lr_decay_epoch = [8, 16, 24, 32, 40]
_C.lr_decay_ratio = 0.1
_C.num_workers = 8
_C.num_epoch = 20
_C.device = 'cuda'
_C.view_freq = 20
_C.view_valid_freq = 1

# ---------------------------------------------------------------------------- #
# Network definition
# ---------------------------------------------------------------------------- #

_N = CN()

# --------------------------------------------
# Mobile Residual Network (MobileResNet) configurations
# --------------------------------------------
_N.MobileResNet34 = CN()
_N.MobileResNet34.network_name = 'MobileResNet34'
_N.MobileResNet34.input_opt = 'Stem-S' # option: 'O' or 'L'
_N.MobileResNet34.output_opt = 'E' # option: 'O' or 'E'
_N.MobileResNet34.block_opt = 'IR'

_N.MobileResNet50 = edict()
_N.MobileResNet50.network_name = 'MobileResNet50'
_N.MobileResNet50.input_opt = 'Stem-S' # option: 'O' or 'L'
_N.MobileResNet50.output_opt = 'E' # option: 'O' or 'E'
_N.MobileResNet50.block_opt = 'IR'

# --------------------------------------------
# Residual Network (ResNet) configurations
# --------------------------------------------
_N.ResNet18 = edict()
_N.ResNet18.network_name = 'ResNet18'
_N.ResNet18.input_opt = 'L' # option: 'O' or 'L' # what is 'DS' ?
_N.ResNet18.output_opt = 'E' # option: 'O' or 'E'
_N.ResNet18.block_opt = 'IR' # option: 'O' or 'IR'
_N.ResNet18.use_se = False

# ---------------------------------------------------------------------------- #
# Loss definition
# ---------------------------------------------------------------------------- #

_L = CN()

# --------------------------------------------
# MagFace loss configurations
# --------------------------------------------
_L.MagFace = CN()
_L.MagFace.loss_name = 'ArcFace'
_L.MagFace.emd_size = 512
_L.MagFace.loss_s = 64.0
_L.MagFace.loss_m = 0.5
_L.MagFace.easy_margin = False

# --------------------------------------------
# Circle loss configurations
# --------------------------------------------
_L.Circle_loss = CN()
_L.Circle_loss.loss_name = 'Circle_loss'
## circle not complete ! ##