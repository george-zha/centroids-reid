import logging
import os
from time import time
import torch
import torch_neuron
import sys
import numpy as np

sys.path.append(".")

from datasets.transforms import ReidTransforms
from config import cfg
from train_ctl_model import CTLModel
from utils.reid_metric import get_dist_func
from inference_utils import (
    ImageDataset,
    ImageFolderWithPaths,
    calculate_centroids,
    create_pid_path_index,
    make_inference_data_loader,
    run_inference,
)

CONFIG = '/home/ubuntu/centroids-reid/configs/256_resnet50_inference.yml'
QUERY_DATA = '/home/ubuntu/datasets/inference/query/'
GALLERY_DATA = '/home/ubuntu/datasets/inference/gallery/'

### Prepare logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)

cfg.merge_from_file(CONFIG)

### Data preparation

log.info(f"Preparing query and gallery dataloaders")
gallery_loader = make_inference_data_loader(cfg, GALLERY_DATA, ImageDataset)
query_loader = make_inference_data_loader(cfg, QUERY_DATA, ImageDataset)

if len(gallery_loader) == 0 or len(query_loader) == 0:
    raise RuntimeError("Length of dataloader = 0")

## Load model
model_neuron = torch.jit.load(cfg.MODEL.PRETRAIN_PATH)

### Inference
latency = []
log.info("Running inference on gallery")
gallery_embeddings = []
while len(gallery_embeddings) < 100:
    for img in gallery_loader:
        delta_start = time()
        embedding = model_neuron(img[0])
        delta = time()-delta_start
        latency.append(delta)
        gallery_embeddings.append(embedding)
print("number of iterations: " + str(len(gallery_embeddings)))
print("Latency P50: {:.0f}".format(np.percentile(latency, 50)*1000.0))
print("Latency P90: {:.0f}".format(np.percentile(latency, 90)*1000.0))
print("Latency P95: {:.0f}".format(np.percentile(latency, 95)*1000.0))
print("Latency P99: {:.0f}\n".format(np.percentile(latency, 99)*1000.0))
