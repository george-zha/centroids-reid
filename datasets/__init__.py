# encoding: utf-8
"""
Partially based on work by:
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com

Adapted and extended by:
@author: mikwieczorek
"""

from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501, VerkadaData, CombinedData
from .df1 import DF1
from .street2shop import Street2Shop
from .lpw import LPW
from .bases import ReidBaseDataModule

__factory = {
    "market1501": Market1501,
    "dukemtmcreid": DukeMTMCreID,
    "df1": DF1,
    "street2shop": Street2Shop,
    "verkada_data": VerkadaData,
    "combined_data": CombinedData,
    "lpw": LPW,
}


def get_names():
    return __factory.keys()


def init_dataset(dataset_names, *args, **kwargs):
    data_object = ReidBaseDataModule(*args, **kwargs)
    datasets = []
    for name in dataset_names:
        if name not in __factory.keys():
            raise KeyError("Unknown datasets: {}".format(name))
        datasets.append(__factory[name](**kwargs))
    data_object.datasets = datasets
    return data_object
