import os.path as osp
from collections import defaultdict
from os import listdir


class LPW():
    """
    Labelled Pedestrian in the Wild
    Reference:
    Song et al. Region-based Quality Estimation Network for Large-Scale Person Re-identification. AAAI 2018.
    URL: https://liuyu.us/dataset/lpw/index.html

    Dataset statistics:
    # identities: 2731, (1975 train, 756 test)
    # images: 77 frames average per identity
    """
    dataset_dir = 'lpw'

    def __init__(self, cfg, **kwargs):
        self.dataset_dir = osp.join(cfg.DATASETS.ROOT_DIR, self.dataset_dir)
        self.query_dir = [[osp.join(self.dataset_dir, 'scen1', 'view2')]]
        self.gallery_dir = [[osp.join(self.dataset_dir, 'scen1', 'view1'), osp.join(self.dataset_dir, 'scen1', 'view3')]]
        self.train_dir = []

        for scene in ['scen2', 'scen3']:
            self.train_dir.append([])
            for view in listdir(osp.join(self.dataset_dir, scene)):
                self.train_dir[-1].append(osp.join(self.dataset_dir, scene, view))


    def _process_dir(self, dir_path, reindex=0, relabel=False):
        dataset_dict = defaultdict(list)
        dataset = []
        label2pid = {}
        lastscenepid, pid, idx = 0, 0, 0

        for scene in dir_path:
            camid = 1 if dir_path == self.gallery_dir else 0
            lastscenepid += max(label2pid.keys()) if label2pid.keys() else 0
            for view in scene:
                for pidfolder in listdir(view):
                    label = int(pidfolder) + lastscenepid

                    if label not in label2pid and relabel:
                        label2pid[label] = pid + reindex
                        pid += 1
                    elif label not in label2pid:
                        label2pid[label] = label + reindex
                    pidc = label2pid[label]

                    for image in listdir(osp.join(view, pidfolder)):
                        img_path = osp.join(view, pidfolder, image)
                        dataset.append((img_path, pidc, camid, idx))
                        dataset_dict[pidc].append((img_path, pidc, camid, idx))
                        idx += 1
                camid += 1
            
        return dataset, dataset_dict
