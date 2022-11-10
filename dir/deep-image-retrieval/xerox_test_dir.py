import sys
import os
import os.path as osp
import pdb
import io
import json
import tqdm
import numpy as np
import torch
import torch.nn.functional as F

# from dirtorch.utils.convenient import mkdir
from dirtorch.utils import common
# from dirtorch.utils.common import tonumpy, matmul, pool
# from dirtorch.utils.pytorch_loader import get_loader
import dirtorch.nets as nets

import pickle as pkl
import hashlib


# Listing available architectures:
#         resnet101_fpn0_rmac
#         resnet101
#         resnet50
#         resnet50_rmac
#         resnet152_fpn_rmac
#         resnet101_rmac
#         resnet152
#         resnet101_fpn_rmac
#         resnet18
#         resnet18_rmac
#         resnet18_fpn_rmac
#         resnet50_fpn_rmac
#         resnet152_rmac
def load_model(path, iscuda):
    checkpoint = common.load_checkpoint(path, iscuda)
    net = nets.create_model(pretrained="", **checkpoint['model_options'])
    net = common.switch_model_to_cuda(net, iscuda, checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.preprocess = checkpoint.get('preprocess', net.preprocess)
    if 'pca' in checkpoint:
        net.pca = checkpoint.get('pca')
    return net


if __name__=="__main__":
    checkpoint = "D:/ORG India/Image-Retreival/dir/deep-image-retrieval/dirtorch/models/Resnet-101-AP-GeM.pt"
    net = torch.load(checkpoint)
    print(net)