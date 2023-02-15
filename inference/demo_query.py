import logging
import os
from time import time
import torch
import sys
import numpy as np
import ipdb

sys.path.append(".")

from config import cfg
from train_ctl_model import CTLModel
from inference_utils import (
    ImageDataset,
    make_inference_data_loader,
    _inference,
)

CONFIG = '/home/georgez/centroids-reid/configs/256_resnet50_inference.yml'
GALLERY_DATA = '/home/georgez/datasets/market1501/bounding_box_test/'\


extract_func = (
    lambda x: (x).rsplit(".", 1)[0].rsplit("_", 1)[0]
)

def calculate_embeddings(cfg, model, dataloader):
    embeddings = [None for i in range(100)]
    paths = [None for i in range(100)]
    latency = []

    for idx, elem in enumerate(dataloader):
        if idx == 100:
            log.info(f"Number of processed images: {idx}")
            break
        delta_start = time()
        embeddings[idx], paths[idx] = _inference(model, elem, False, cfg.TEST.FEAT_NORM)
        delta = time()
        latency.append(delta-delta_start)
        ipdb.set_trace()

    print("number of iterations: " + str(len(embeddings)))
    print("Latency P50: {:.0f}".format(np.percentile(latency, 50)*1000.0))
    print("Latency P90: {:.0f}".format(np.percentile(latency, 90)*1000.0))
    print("Latency P95: {:.0f}".format(np.percentile(latency, 95)*1000.0))
    print("Latency P99: {:.0f}\n".format(np.percentile(latency, 99)*1000.0))

    return embeddings, paths

def calculate_centroids(embeddings, paths):
    pid2paths = {}
    pid2ind = {}
    for idx, path in enumerate(paths):
        ipdb.set_trace()
        pid = extract_func(path)
        if pid not in pid2paths:
            pid2paths[pid] = [path]
            pid2ind[pid] = [idx]
        else:
            pid2paths[pid].append(path)
            pid2ind[pid].append(idx)

    centroids = []
    pids_centroids_inds = []
    for pid, indices in pid2ind.items():
        inds = np.array(indices)
        pids_vecs = embeddings[inds]
        length = pids_vecs.shape[0]
        centroid = np.sum(pids_vecs, 0) / length
        pids_centroids_inds.append(pid)
        centroids.append(centroid)
    centroids_arr = np.vstack(np.array(centroids))
    pids_centroids_inds = np.array(pids_centroids_inds, dtype=np.str_)
    return centroids_arr, pids_centroids_inds, pid2paths

### Prepare logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)

cfg.merge_from_file(CONFIG)

### Data preparation

log.info(f"Preparing gallery data")
gallery_loader = make_inference_data_loader(cfg, GALLERY_DATA, ImageDataset)
if len(gallery_loader) == 0:
    raise RuntimeError("Length of dataloader = 0")

device = torch.device('cpu')
model = CTLModel(cfg=cfg, num_classes=751)
model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH, map_location=device))

embeddings, paths = calculate_embeddings(cfg, model, gallery_loader)

centroids, pid2cind, pid2paths = calculate_centroids(embeddings, paths)

import ipdb
ipdb.set_trace()