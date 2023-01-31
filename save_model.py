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

PATH = "resnet50_model.pt"
cfg.MODEL.PRETRAIN_PATH = '/home/george/centroids-reid/epoch=119.ckpt'
model = CTLModel.load_from_checkpoint(cfg.MODEL.PRETRAIN_PATH)
torch.save(model.state_dict(), PATH)
