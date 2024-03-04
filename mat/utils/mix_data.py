import numpy as np
import functools
import math
import torch
import pickle
import os

import pysc2

expert_data_path = "/home/LAB/qiuyue/Multi-Agent-Transformer-main/mat/scripts/"\
                   "save_data/smac/6h_vs_8z_mat_5w_expert/StarCraft2.pkl"
medium_data_path = "/home/LAB/qiuyue/Multi-Agent-Transformer-main/mat/scripts/"\
                   "save_data/smac/6h_vs_8z_mat_5w_medium/StarCraft2.pkl"
with open(expert_data_path, 'rb') as f:
    expert_set = pickle.load(f)
with open(medium_data_path, 'rb') as f:
    medium_set = pickle.load(f)
expert_set.extend(medium_set)

medium_data_dir = "/home/LAB/qiuyue/Multi-Agent-Transformer-main/mat/scripts/"\
                   "save_data/smac/6h_vs_8z_mat_5w_mix/StarCraft2.pkl"
if not os.path.isdir(medium_data_dir):
    os.makedirs(medium_data_dir)
with open(medium_data_dir + '/StarCraft2.pkl', 'wb') as f:
    pickle.dump(expert_set, f)
print(len(expert_set))

