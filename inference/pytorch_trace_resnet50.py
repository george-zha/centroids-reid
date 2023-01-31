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
from inference.inference_utils import (
    ImageDataset,
    make_inference_data_loader,
)
from train_ctl_model import CTLModel

PATH = '/home/ubuntu/centroids-reid/model.pt'

image = torch.zeros([3,256,128], dtype=torch.float32)
image = ([3,256,128])

image = (torch.zeros([3,256,128], dtype=torch.float32), torch.tensor(0), torch.tensor(0))

## Load a pretrained ResNet50 model
device = torch.device('cpu')
cfg.merge_from_file('/home/ubuntu/centroids-reid/configs/256_resnet50.yml')
cfg.TEST.ONLY_TEST = True
cfg.DATASETS.ROOT_DIR = '/home/ubuntu/data/'
cfg.TEST.IMS_PER_BATCH = 1
val_loader = make_inference_data_loader(cfg, cfg.DATASETS.ROOT_DIR, ImageDataset)
image = next(iter(val_loader))

model = CTLModel(cfg=cfg, test_dataloader=None, num_query=751, num_classes=751)
model.load_state_dict(torch.load(PATH, map_location=device))


#model = CTLModel.load_from_checkpoint(cfg.MODEL.PRETRAIN_PATH)

## Tell the model we are using it for evaluation (not training)
model.eval()
model_neuron = torch.neuron.trace(model, example_inputs=[image])

## Export to saved model
model_neuron.save("resnet50_neuron.pt")