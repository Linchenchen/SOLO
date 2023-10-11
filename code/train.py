import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
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

batch_size = 12
train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
train_loader = train_build_loader.loader()
test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = test_build_loader.loader()

# build the model
resnet50_fpn = Resnet50Backbone()
solo_head = SOLOHead(num_classes=4) ## class number is 4, because consider the background as one category.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet50_fpn.to(device)
solo_head = solo_head.to(device)

num_epochs = 36
# The SGD optimizer with a weight decay of 1e-4 and a momentum of 0.9 is used. 
# For a batch size of 16, an initial learning rate of 0.01 is used (this should be scaled appropriately for different batch sizes).
optimizer = optim.SGD(solo_head.parameters(), lr=0.01/16*batch_size, momentum=0.9, weight_decay=0.0001)
# reducing the learning rate by a factor of 10 at epochs 27 and 33
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[27,33], gamma=0.1)

os.makedirs("checkpoints", exist_ok=True)
checkpoint_dir = 'checkpoints/'

# loop the image
train_cate_losses=[]
train_mask_losses=[]
train_total_losses=[]
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    ## fill in your training code
    solo_head.train()
    running_cate_loss = 0.0
    running_mask_loss = 0.0
    running_total_loss= 0.0

    for iter, data in enumerate(train_loader, 0):
        img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
        img = img.to(device)
        label_list = [label.to(device) for label in label_list]
        mask_list = [mask.to(device) for mask in mask_list]
        bbox_list = [bbox.to(device) for bbox in bbox_list]

        # fpn is a dict
        with torch.no_grad():
            backout = resnet50_fpn(img)
        del img
        fpn_feat_list = list(backout.values())
        
        # make the target
        optimizer.zero_grad()
        cate_pred_list, ins_pred_list = solo_head(fpn_feat_list, eval=False)

        del fpn_feat_list
        ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                        bbox_list,
                                                                        label_list,
                                                                        mask_list)
        mask_color_list = ["jet", "ocean", "Spectral"]
        # solo_head.PlotGT(ins_gts_list,ins_ind_gts_list,cate_gts_list,mask_color_list,img,iter)

        cate_loss, mask_loss, total_loss=solo_head.loss(cate_pred_list,ins_pred_list,
                                                        ins_gts_list,ins_ind_gts_list,cate_gts_list)
        
        del label_list, mask_list, bbox_list
        del ins_gts_list, ins_ind_gts_list, cate_gts_list, cate_pred_list, ins_pred_list      

        total_loss.backward()
        optimizer.step()
        running_cate_loss += cate_loss.item()
        running_mask_loss += mask_loss.item()
        running_total_loss += total_loss.item()

        if iter % 100 == 99:
            log_cate_loss = running_cate_loss / (iter + 1)
            log_mask_loss = running_mask_loss / (iter + 1)
            log_total_loss = running_total_loss / (iter + 1)
        
            train_cate_losses.append(log_cate_loss)
            train_mask_losses.append(log_mask_loss)
            train_total_losses.append(log_total_loss)

            print('\nIteration:{} Avg. train total loss: {:.4f}'.format(iter+1, log_total_loss))
     
    # Calculate and save average loss or metrics for this epoch
    avg_cate_loss = running_cate_loss / (iter + 1)
    avg_mask_loss = running_mask_loss / (iter + 1)
    avg_total_loss = running_total_loss / (iter + 1)

    running_cate_loss = 0.0
    running_mask_loss = 0.0
    running_total_loss = 0.0

    # Append to the list of losses or metrics
    train_losses.append((avg_cate_loss, avg_mask_loss, avg_total_loss))

    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Avg. train cate loss: {avg_cate_loss:.4f}, '
          f'Avg. train mask loss: {avg_mask_loss:.4f}, '
          f'Avg. train total loss: {avg_total_loss:.4f}')
    
    # save the model checkpoint
    checkpoint_filename = f'{checkpoint_dir}model_epoch_{epoch}.pt'
    torch.save({'epoch': epoch,
                'model_state_dict': solo_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                }, checkpoint_filename)
    
    solo_head.eval()

    test_running_cate_loss = 0.0    
    test_running_mask_loss=0.0
    test_running_total_loss=0.0
    
    with torch.no_grad():
        for iter, data in enumerate(test_loader, 0):   
            img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
            img = img.to(device)
            label_list = [label.to(device) for label in label_list]
            mask_list = [mask.to(device) for mask in mask_list]
            bbox_list = [bbox.to(device) for bbox in bbox_list]

            backout = resnet50_fpn(img)
            del img
            fpn_feat_list = list(backout.values())

            cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False) 
            del fpn_feat_list
            ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                    bbox_list,
                                                                    label_list,
                                                                    mask_list)

            cate_loss, mask_loss, total_loss=solo_head.loss(cate_pred_list,ins_pred_list,ins_gts_list,
                                                            ins_ind_gts_list,cate_gts_list)
            
            test_running_cate_loss += cate_loss.item()
            test_running_mask_loss += mask_loss.item()
            test_running_total_loss += total_loss.item()
            
            del ins_gts_list, ins_ind_gts_list, cate_gts_list, cate_pred_list, ins_pred_list
            
        epoch_cate_loss = test_running_cate_loss / (iter+1)
        epoch_mask_loss = test_running_mask_loss / (iter+1)
        epoch_total_loss = test_running_total_loss / (iter+1)
        print('\nEpoch:{} Avg. test loss: {:.4f}\n'.format(epoch + 1, epoch_total_loss))
        test_losses.append((epoch_cate_loss, epoch_mask_loss, epoch_total_loss))
    
    scheduler.step()
