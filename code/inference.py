import os
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import ndimage

from dataset import *
from backbone import *
from solo_head import *

# file path and make a list
imgs_path = '../data/hw3_mycocodata_img_comp_zlib.h5'
masks_path = '../data/hw3_mycocodata_mask_comp_zlib.h5'
labels_path = "../data/hw3_mycocodata_labels_comp_zlib.npy"
bboxes_path = "../data/hw3_mycocodata_bboxes_comp_zlib.npy"
paths = [imgs_path, masks_path, labels_path, bboxes_path]
# load the data into data.Dataset
dataset = BuildDataset(paths)

## Visualize debugging
# --------------------------------------------
# build the dataloader
# set 20% of the dataset as the training data
full_size = len(dataset)
train_size = int(full_size * 0.8)
test_size = full_size - train_size
# random split the dataset into training and testset
# set seed
torch.random.manual_seed(1)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
# push the randomized training data into the dataloader

batch_size = 2
train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
train_loader = train_build_loader.loader()
test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = test_build_loader.loader()

resnet50_fpn = Resnet50Backbone()
solo_head = SOLOHead(num_classes=4) ## class number is 4, because consider the background as one category.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet50_fpn.to(device)
solo_head = solo_head.to(device)

# The SGD optimizer with a weight decay of 1e-4 and a momentum of 0.9 is used. 
# For a batch size of 16, an initial learning rate of 0.01 is used (this should be scaled appropriately for different batch sizes).
optimizer = optim.SGD(solo_head.parameters(), lr=0.01/16*batch_size, momentum=0.9, weight_decay=0.0001)
# reducing the learning rate by a factor of 10 at epochs 27 and 33
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[27,33], gamma=0.1)

checkpoint = torch.load('../checkpoints/model_epoch_30.pt', map_location=torch.device('cpu'))
solo_head.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
train_losses = checkpoint['train_losses']

solo_head.eval()

with torch.no_grad():
    for iter, data in enumerate(test_loader, 0):
        img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
        img = img.to(device)
        label_list = [label.to(device) for label in label_list]
        mask_list = [mask.to(device) for mask in mask_list]
        bbox_list = [bbox.to(device) for bbox in bbox_list]
        # fpn is a dict
        backout = resnet50_fpn(img)
        fpn_feat_list = list(backout.values())
        # make the target

        ## demo
        cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval = True)
        # ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
        #                                                                   bbox_list,
        #                                                                   label_list,
        #                                                                   mask_list)
        
        mask_color_list = ["jet", "ocean", "Spectral"]

        # solo_head.PlotGT(ins_gts_list,ins_ind_gts_list,cate_gts_list,mask_color_list,img,iter)

        NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list = solo_head.PostProcess(ins_pred_list,cate_pred_list,(img.shape[2],img.shape[3]))
        solo_head.PlotInfer(NMS_sorted_scores_list,NMS_sorted_cate_label_list,NMS_sorted_ins_list,mask_color_list,img,iter)
        
        if iter == 20:
            break
