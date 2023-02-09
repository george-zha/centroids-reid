import json
import os
import random
import cv2 as cv
from collections import defaultdict

public = '/home/george/datasets/market1501/'
verkada = '/home/george/datasets/verkada_data_copy/'
combined = '/home/george/datasets/combined_data/'

os.mkdir(combined)

for type in ['bounding_box_test/', 'bounding_box_train/', 'query/']:
    os.mkdir(combined + type)

    for filepath in os.listdir(public + type):
        os.system('cp ' + public + type + filepath + ' ' + combined + type + filepath)
    for filepath in os.listdir(verkada + type):
        orig = filepath
        filepath = filepath.split('_')
        filepath[0] = str(int(filepath[0]) + 1501)
        filepath = '_'.join(filepath)
        os.system('cp ' + verkada + type + orig + ' ' + combined + type + filepath)

