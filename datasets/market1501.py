# encoding: utf-8
"""
Partially based on work by:
@author:  sherlock
@contact: sherlockliao01@gmail.com

Adapted and extended by:
@author: mikwieczorek
"""

import glob
import os.path as osp
import re
from collections import defaultdict


class Market1501():
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)

    Version that will not supply resampled instances
    """
    dataset_dir = 'market1501'

    def __init__(self, cfg, **kwargs):
        self.dataset_dir = osp.join(cfg.DATASETS.ROOT_DIR, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

    def _process_dir(self, dir_path, relabel=0, camlabel = 0):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: 
                img_paths.remove(img_path)
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label+relabel for label, pid in enumerate(pid_container)}

        dataset_dict = defaultdict(list)
        dataset = []

        for idx, img_path in enumerate(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue
            assert 1 <= camid <= 6
            camid = camid + camlabel - 1  # index starts from 0
            pid = pid2label[pid]
            dataset.append((img_path, pid, camid, idx))
            dataset_dict[pid].append((img_path, pid, camid, idx))

        return dataset, dataset_dict


class VerkadaData(Market1501):
    dataset_dir = 'verkada_data'

    def _process_dir(self, dir_path, relabel=0, camlabel=0):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label+relabel for label, pid in enumerate(pid_container)}

        dataset_dict = defaultdict(list)
        dataset = []

        for idx, img_path in enumerate(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            camid = camid + camlabel -1 # index starts from 0
            pid = pid2label[pid]
            dataset.append((img_path, pid, camid, idx))
            dataset_dict[pid].append((img_path, pid, camid, idx))

        return dataset, dataset_dict 


class CombinedData(VerkadaData):
    dataset_dir = 'combined_data'
    