# encoding: utf-8
"""
@author: mikwieczorek
"""

import copy
import random
import torch
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances, dataset_sizes, world_size=None, rank=None):
        self.data_source = data_source
        self.batch_size = batch_size  # This is the number of unique pids in reality
        self.dataset_sizes = dataset_sizes  # This is the number of pids for each dataset per batch
        self.num_instances = num_instances
        self.index_dic = [defaultdict(list) for i in self.data_source]
        self.pids = set()
        lens_list = []
        
        for i, dataset in enumerate(self.data_source):
            for pid, idxs in dataset.items():
                len_ = len(idxs)
                if len_ % self.num_instances == 1:
                    len_ -= 1
                pid_full_ocurances= int(np.ceil(len_/self.num_instances))
                # index_dic tracks how many samples can be taken from a pid based on num_instances and num of items in that pid
                self.index_dic[i][pid].extend([pid] * pid_full_ocurances)
                lens_list.append(pid_full_ocurances)
            self.pids.update(self.index_dic[i].keys())

        self.length = sum(lens_list)

        self.world_size = world_size
        self.rank = rank
        self.epoch = 0  # For deteministic pids shuffling

        self.length = self.length // self.world_size


    def __iter__(self):
         # deterministically shuffle based on epoch
        np.random.seed(self.epoch)
        random.seed(self.epoch)  # Just in case ...

        batch_idxs_datasets = copy.deepcopy(self.index_dic)
        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        # Ensure we iterate through all pids before exiting
        while len(avai_pids) >= self.batch_size * self.world_size:

            for i,dataset_size in enumerate(self.dataset_sizes):
                batch_idxs_dict = batch_idxs_datasets[i]
                selected_pids = random.sample(batch_idxs_dict.keys(), dataset_size * self.world_size)

                for pid in selected_pids:
                    batch_idxs = batch_idxs_dict[pid].pop(0)
                    final_idxs.append(batch_idxs)
                    if len(batch_idxs_dict[pid]) == 0:
                        batch_idxs_dict.pop(pid)
                        avai_pids.discard(pid)
                
                # Reuse pids from dataset if dataset empties but other datasets still have unseen pids
                if len(batch_idxs_dict) < dataset_size * self.world_size:
                    batch_idxs_datasets[i] = copy.deepcopy(self.index_dic[i])
                    
        assert len(final_idxs) % (self.batch_size * self.world_size) == 0, f"Number of elements in the sampler indices {len(final_idxs)} must be divisible by the batch_size {self.batch_size * self.world_size}, but it is not!"
        # final_idxs in groups of batch_size * world_size, randomly allocated, split by world_size and distribute by rank
        final_idxs = list(np.array_split(final_idxs, self.world_size)[self.rank])

        if len(final_idxs) % self.batch_size != 0:
            reminder = len(final_idxs) % self.batch_size
            final_idxs = final_idxs[:-reminder]

        assert len(final_idxs) % self.batch_size == 0, f"Number of elements in the sampler indices after the split {len(final_idxs)} must be divisible by the batch_size {self.batch_size}, but it is not!"
        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epoch = epoch
