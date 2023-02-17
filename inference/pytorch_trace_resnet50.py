import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import os
import torch_neuron

sys.path.append(".")

from config import cfg
from train_ctl_model import CTLModel

MODEL = '/home/ubuntu/centroids-reid/models/verkada-model-resnet50.pt'
CONFIG = '/home/ubuntu/centroids-reid/configs/256_resnet50_inference.yml'

## Load a pretrained ResNet50 model

cfg.merge_from_file(CONFIG)
cfg.TEST.ONLY_TEST = True

image = torch.zeros([cfg.TEST.IMS_PER_BATCH, 3] + cfg.INPUT.SIZE_TEST, dtype=torch.float32)

device = torch.device('cpu')
model = CTLModel(cfg=cfg, num_classes=14778)
model.load_state_dict(torch.load(MODEL, map_location=device))

## Tell the model we are using it for evaluation (not training)
model.eval()
model_neuron = torch.neuron.trace(model, example_inputs=[image], dynamic_batch_size=True)

## Export to saved model
model_neuron.save("/home/ubuntu/centroids-reid/models/verkada-neuron-resnet50.pt")