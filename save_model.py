import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(".")

from config import cfg
from train_ctl_model import CTLModel

PATH = "/home/georgez/centroids-reid/experiments/verkada-model-resnet50.pt"
cfg.MODEL.PRETRAIN_PATH = '/home/georgez/centroids-reid/logs/verkada_data/train_ctl_model/version_18/checkpoints/last.ckpt'
model = CTLModel.load_from_checkpoint(cfg.MODEL.PRETRAIN_PATH)
torch.save(model.state_dict(), PATH)
