import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from dataset import *
from functools import partial
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning)

class SOLOHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels=256,
                 seg_feat_channels=256,
                 stacked_convs=7,
                 strides=[8, 8, 16, 32, 32],
                 scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
                 epsilon=0.2,
                 num_grids=[40, 36, 24, 16, 12],
                 cate_down_pos=0,
                 with_deform=False,
                 mask_loss_cfg=dict(weight=3),
                 cate_loss_cfg=dict(gamma=2,
                                alpha=0.25,
                                weight=1),
                 postprocess_cfg=dict(cate_thresh=0.2,
                                      ins_thresh=0.5,
                                      pre_NMS_num=50,
                                      keep_instance=5,
                                      IoU_thresh=0.5)):
        super(SOLOHead, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.epsilon = epsilon
        self.cate_down_pos = cate_down_pos
        self.scale_ranges = scale_ranges
        self.with_deform = with_deform

        self.mask_loss_cfg = mask_loss_cfg
        self.cate_loss_cfg = cate_loss_cfg
        self.postprocess_cfg = postprocess_cfg
        # initialize the layers for cate and mask branch, and initialize the weights
        self._init_layers()
        self._init_weights()

        # check flag
        assert len(self.ins_head) == self.stacked_convs
        assert len(self.cate_head) == self.stacked_convs
        assert len(self.ins_out_list) == len(self.strides)
        pass

    # This function build network layer for cate and ins branch
    # it builds 4 self.var
        # self.cate_head is nn.ModuleList 7 inter-layers of conv2d
        # self.ins_head is nn.ModuleList 7 inter-layers of conv2d
        # self.cate_out is 1 out-layer of conv2d
        # self.ins_out_list is nn.ModuleList len(self.seg_num_grids) out-layers of conv2d, one for each fpn_feat
    def _init_layers(self):
        ## TODO initialize layers: stack intermediate layer and output layer
        # define groupnorm
        num_groups = 32
        # initial the two branch head modulelist
        self.cate_head = nn.ModuleList()
        self.ins_head = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            self.ins_head.append(
                nn.Sequential(
                    nn.Conv2d(
                        chn,
                        self.seg_feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        bias=False),

                    nn.GroupNorm(num_channels=self.seg_feat_channels,
                    num_groups=num_groups),
                    
                    nn.ReLU(inplace=True)
                )
            )

            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_head.append(nn.Sequential(
                nn.Conv2d(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    bias=False),

                nn.GroupNorm(num_channels=self.seg_feat_channels,
                num_groups=num_groups),

                nn.ReLU(inplace=True)
                )
            )

        self.ins_out_list = nn.ModuleList()
        for seg_num_grid in self.seg_num_grids:
            self.ins_out_list.append(nn.Sequential(
                nn.Conv2d(self.seg_feat_channels, seg_num_grid**2, 1),
                nn.Sigmoid()
                )
            )

        self.cate_out = nn.Sequential(
            nn.Conv2d(self.seg_feat_channels, self.cate_out_channels, 3, padding=1),
            nn.Sigmoid() 
        )

    # This function initialize weights for head network
    def _init_weights(self):
        
        # def normal_init(module, mean=0, std=1, bias=0):
        #     nn.init.normal_(module.weight, mean, std)
        #     if hasattr(module, 'bias') and module.bias is not None:
        #         nn.init.constant_(module.bias, bias)

        # def bias_init_with_prob(prior_prob):
        #     """initialize conv/fc bias value according to giving probablity."""
        #     bias_init = float(-np.log((1 - prior_prob) / prior_prob))
        #     return bias_init

        ## TODO: initialize the weights
        # for m in self.ins_head:
        #     if isinstance(m, nn.Sequential):
        #         for con in m:
        #             if isinstance(con, nn.Conv2d):
        #                 normal_init(con, std=0.01)
        # for m in self.cate_head:
        #     if isinstance(m, nn.Sequential):
        #         for con in m:
        #             if isinstance(con, nn.Conv2d):
        #                 normal_init(con, std=0.01)

        # bias_ins = bias_init_with_prob(0.01)
        # for m in self.ins_out_list: 
        #     normal_init(m, std=0.01, bias=bias_ins)

        # bias_cate = bias_init_with_prob(0.01)
        # normal_init(self.cate_out, std=0.01, bias=bias_cate)
        for m in self.cate_head.modules():
            if isinstance(m, nn.Conv2d): 
                m.weight.data =torch.nn.init.xavier_uniform_(m.weight.data)
                assert m.bias is None
        for m in self.ins_head.modules():
            if isinstance(m, nn.Conv2d): 
                m.weight.data =torch.nn.init.xavier_uniform_(m.weight.data)
                assert m.bias is None
        for m in self.cate_out.modules():
            if isinstance(m, nn.Conv2d): 
                m.weight.data =torch.nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias, 0) 
        for layer in self.ins_out_list:
            for m in layer:
                if isinstance(m, nn.Conv2d): 
                    m.weight.data =torch.nn.init.xavier_uniform_(m.weight.data) 
                    nn.init.constant_(m.bias, 0)

    # Forward function should forward every levels in the FPN.
    # this is done by map function or for loop
    # Input:
        # fpn_feat_list: backout_list of resnet50-fpn
    # Output:
        # if eval = False
            # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
            # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # if eval==True
            # cate_pred_list: list, len(fpn_level), each (bz,S,S,C-1) / after point_NMS
            # ins_pred_list: list, len(fpn_level), each (bz, S^2, Ori_H, Ori_W) / after upsampling
    def forward(self,
                fpn_feat_list,
                eval=False):
        new_fpn_list = self.NewFPN(fpn_feat_list)  # stride[8,8,16,32,32]
        assert new_fpn_list[0].shape[1:] == (256,100,136)
        quart_shape = [new_fpn_list[0].shape[-2]*2, new_fpn_list[0].shape[-1]*2]  # stride: 4
        # TODO: use MultiApply to compute cate_pred_list, ins_pred_list. Parallel w.r.t. feature level.
        cate_pred_list, ins_pred_list = self.MultiApply(self.forward_single_level,
                                                        new_fpn_list,
                                                        list(range(len(self.seg_num_grids))),
                                                        eval=eval,
                                                        upsample_shape=quart_shape)
        assert len(new_fpn_list) == len(self.seg_num_grids)

        # assert cate_pred_list[1].shape[1] == self.cate_out_channels
        assert ins_pred_list[1].shape[1] == self.seg_num_grids[1]**2
        assert cate_pred_list[1].shape[2] == self.seg_num_grids[1]
        return cate_pred_list, ins_pred_list

    # This function upsample/downsample the fpn level for the network
    # In paper author change the original fpn level resolution
    # Input:
        # fpn_feat_list, list, len(FPN), stride[4,8,16,32,64]
    # Output:
    # new_fpn_list, list, len(FPN), stride[8,8,16,32,32]
    def NewFPN(self, fpn_feat_list):
        return (F.interpolate(fpn_feat_list[0], scale_factor=0.5, recompute_scale_factor=True),
                fpn_feat_list[1],
                fpn_feat_list[2],
                fpn_feat_list[3],
                F.interpolate(fpn_feat_list[4], size=fpn_feat_list[3].shape[-2:]))


    # This function forward a single level of fpn_featmap through the network
    # Input:
        # fpn_feat: (bz, fpn_channels(256), H_feat, W_feat)
        # idx: indicate the fpn level idx, num_grids idx, the ins_out_layer idx
    # Output:
        # if eval==False
            # cate_pred: (bz,C-1,S,S)
            # ins_pred: (bz, S^2, 2H_feat, 2W_feat)
        # if eval==True
            # cate_pred: (bz,S,S,C-1) / after point_NMS
            # ins_pred: (bz, S^2, Ori_H/4, Ori_W/4) / after upsampling
    def forward_single_level(self, fpn_feat, idx, eval=False, upsample_shape=None):
        # upsample_shape is used in eval mode
        ## TODO: finish forward function for single level in FPN.
        ## Notice, we distinguish the training and inference.
        cate_pred = fpn_feat
        ins_pred = fpn_feat
        num_grid = self.seg_num_grids[idx]  # current level grid

        # concat coord
        x_range = torch.linspace(-1, 1, ins_pred.shape[-1], device=ins_pred.device)
        y_range = torch.linspace(-1, 1, ins_pred.shape[-2], device=ins_pred.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([ins_pred.shape[0], 1, -1, -1])
        x = x.expand([ins_pred.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        ins_pred = torch.cat([ins_pred, coord_feat], 1)

        for i, ins_layer in enumerate(self.ins_head):
            ins_pred = ins_layer(ins_pred)

        ins_pred = F.interpolate(ins_pred, scale_factor=2)
        ins_pred = self.ins_out_list[idx](ins_pred)

        # cate branch
        for i, cate_layer in enumerate(self.cate_head):
            if i == self.cate_down_pos:
                cate_pred = F.interpolate(cate_pred, size=num_grid)
            cate_pred = cate_layer(cate_pred)

        cate_pred = self.cate_out(cate_pred)

        # in inference time, upsample the pred to (ori image size/4)
        if eval == True:
            ## TODO resize ins_pred
            ins_pred = F.interpolate(ins_pred, upsample_shape)
            cate_pred = self.points_nms(cate_pred).permute(0,2,3,1)

        # check flag
        if eval == False:
            assert cate_pred.shape[1:] == (3, num_grid, num_grid)
            assert ins_pred.shape[1:] == (num_grid**2, fpn_feat.shape[2]*2, fpn_feat.shape[3]*2)
        else:
            pass
        return cate_pred, ins_pred

    # Credit to SOLO Author's code
    # This function do a NMS on the heat map(cate_pred), grid-level
    # Input:
        # heat: (bz,C-1, S, S)
    # Output:
        # (bz,C-1, S, S)
    def points_nms(self, heat, kernel=2):
        # kernel must be 2
        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=1)
        keep = (hmax[:, :, :-1, :-1] == heat).float()
        return heat * keep

    # This function compute loss for a batch of images
    # input:
        # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
        # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    # output:
        # cate_loss, mask_loss, total_loss
    def loss(self,
             cate_pred_list,
             ins_pred_list,
             ins_gts_list,
             ins_ind_gts_list,
             cate_gts_list):
        ## TODO: compute loss, vecterize this part will help a lot. 
        # To avoid potential ill-conditioning, if necessary, add a very small number to denominator for focalloss and diceloss computation.
        
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3} -> cate_gts: (bz*fpn*S^2,)
        # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S) -> cate_preds: (bz*fpn*S^2, C-1) 
        cate_gts = [torch.cat([cate_gts_level_img.flatten() for cate_gts_level_img in cate_gts_level]) for cate_gts_level in zip(*cate_gts_list)]  
        cate_gts = torch.cat(cate_gts)  
               
        cate_preds = [cate_pred_level.permute(0,2,3,1).reshape(-1, self.cate_out_channels) for cate_pred_level in cate_pred_list]  
        cate_preds = torch.cat(cate_preds, 0) 
        
        cate_loss= self.FocalLoss(cate_preds, cate_gts)

        # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)

        # list, len(fpn), each(active_across_batch, 2H_feat, 2W_feat)
        ins_gts = [torch.cat([ins_labels_level_img[ins_ind_labels_level_img, ...]
                                 for ins_labels_level_img, ins_ind_labels_level_img in
                                 zip(ins_labels_level, ins_ind_labels_level)], 0)
                      for ins_labels_level, ins_ind_labels_level in zip(zip(*ins_gts_list), zip(*ins_ind_gts_list))]
        ins_preds = [torch.cat([ins_preds_level_img[ins_ind_labels_level_img, ...]
                                for ins_preds_level_img, ins_ind_labels_level_img in zip(ins_preds_level, ins_ind_labels_level)], 0)
                     for ins_preds_level, ins_ind_labels_level in zip(ins_pred_list, zip(*ins_ind_gts_list))]
      
        # dice loss
        loss_dice = []
        for mask_preds, mask_gts in zip(ins_preds, ins_gts):
            if mask_preds.shape[0] == 0:
                continue
            dice_list = map(self.DiceLoss,mask_preds, mask_gts)
            dice_list = list(dice_list)
            dice_avg = torch.stack(dice_list).mean()
            loss_dice.append(dice_avg)
        mask_loss = torch.stack(loss_dice).mean()
        
        total_loss = self.cate_loss_cfg["weight"] * cate_loss + self.mask_loss_cfg["weight"] * mask_loss
        return cate_loss, mask_loss, total_loss



    # This function compute the DiceLoss
    # Input:
        # mask_pred: (2H_feat, 2W_feat)
        # mask_gt: (2H_feat, 2W_feat)
    # Output: dice_loss, scalar
    def DiceLoss(self, mask_pred, mask_gt):
        ## TODO: compute DiceLoss
        dice_loss = 1 - 2 * torch.sum(mask_pred * mask_gt) / (torch.sum(mask_pred**2) + torch.sum(mask_gt**2) + 1e-6)
        return dice_loss

    # This function compute the cate loss
    # Input:
        # cate_preds: (num_entry, C-1)
        # cate_gts: (num_entry,)
    # Output: focal_loss, scalar
    def FocalLoss(self, cate_preds, cate_gts):
        ## TODO: compute focalloss
        alpha = self.cate_loss_cfg['alpha']
        gamma = self.cate_loss_cfg['gamma']
        N = cate_preds.shape[0]
        C = cate_preds.shape[1] + 1 #including background
    
        idx_row=list(np.arange(0,N))
        idx_col=list(cate_gts.long().cpu().numpy())
        cate_gts_one_hot=torch.zeros((N,C), device=cate_preds.device, dtype=torch.long)
        cate_gts_one_hot[idx_row,idx_col] = 1
        cate_gts_one_hot = cate_gts_one_hot[:,1:]

        focal_loss_matrix = - alpha * (1 - cate_preds)**gamma * cate_gts_one_hot * torch.log(cate_preds) \
                    - (1 - alpha) * cate_preds**gamma * (1 - cate_gts_one_hot) * torch.log(1 - cate_preds)
        
        focal_loss = torch.sum(focal_loss_matrix) / (N*C)
        return focal_loss

    def MultiApply(self, func, *args, **kwargs):
        pfunc = partial(func, **kwargs) if kwargs else func
        map_results = map(pfunc, *args)

        return tuple(map(list, zip(*map_results)))

    # This function build the ground truth tensor for each batch in the training
    # Input:
        # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # / ins_pred_list is only used to record feature map
        # bbox_list: list, len(batch_size), each (n_object, 4) (x1y1x2y2 system)
        # label_list: list, len(batch_size), each (n_object, )
        # mask_list: list, len(batch_size), each (n_object, 800, 1088)
    # Output:
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    def target(self,
               ins_pred_list,
               bbox_list,
               label_list,
               mask_list):
        # TODO: use MultiApply to compute ins_gts_list, ins_ind_gts_list, cate_gts_list. Parallel w.r.t. img mini-batch
        # remember, you want to construct target of the same resolution as prediction output in training
        feature_sizes = [(ins_pred.shape[2], ins_pred.shape[3]) for ins_pred in ins_pred_list]
        ins_gts_list, ins_ind_gts_list, cate_gts_list = self.MultiApply(self.target_single_img,
                                                                        bbox_list,
                                                                        label_list,
                                                                        mask_list,
                                                                        featmap_sizes=feature_sizes)
        # check flag
        assert ins_gts_list[0][1].shape == (self.seg_num_grids[1]**2, 200, 272)
        assert ins_ind_gts_list[0][1].shape == (self.seg_num_grids[1]**2,)
        assert cate_gts_list[0][1].shape == (self.seg_num_grids[1], self.seg_num_grids[1])

        return ins_gts_list, ins_ind_gts_list, cate_gts_list
    # -----------------------------------
    ## process single image in one batch
    # -----------------------------------
    # input:
        # gt_bboxes_raw: n_obj, 4 (x1y1x2y2 system)
        # gt_labels_raw: n_obj,
        # gt_masks_raw: n_obj, H_ori, W_ori
        # featmap_sizes: list of shapes of featmap
    # output:
        # ins_label_list: list, len: len(FPN), (S^2, 2H_feat, 2W_feat)
        # cate_label_list: list, len: len(FPN), (S, S)
        # ins_ind_label_list: list, len: len(FPN), (S^2, )
    def target_single_img(self,
                          gt_bboxes_raw,
                          gt_labels_raw,
                          gt_masks_raw,
                          featmap_sizes=None):
        ## TODO: finish single image target build
        # compute the area of every object in this single image
        # initial the output list, each entry for one featmap
        ins_label_list = []
        ins_ind_label_list = []
        cate_label_list = []

        device = gt_bboxes_raw.device

        # compute the area of the object
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        # compute the ins_label, cate_label, ins_ind_label
        for (lower_bound, upper_bound), stride, featmap_size, num_grid \
            in zip(self.scale_ranges, self.strides, featmap_sizes, self.seg_num_grids):

            ins_label = torch.zeros([num_grid ** 2, featmap_size[0], featmap_size[1]], dtype=torch.uint8, device=device)
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            if len(hit_indices) == 0:
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.epsilon
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.epsilon

            # mass center
            gt_masks_pt = gt_masks.to(device=device)

            _, h, w = gt_masks_pt.size()
            ys = torch.arange(0, h, dtype=torch.float32, device=gt_masks_pt.device)
            xs = torch.arange(0, w, dtype=torch.float32, device=gt_masks_pt.device)

            m00 = gt_masks_pt.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
            m10 = (gt_masks_pt * xs).sum(dim=-1).sum(dim=-1)
            m01 = (gt_masks_pt * ys[:, None]).sum(dim=-1).sum(dim=-1)
            center_ws = m10 / m00
            center_hs = m01 / m00

            valid_mask_flags = gt_masks_pt.sum(dim=-1).sum(dim=-1) > 0

            output_stride = stride / 2
            for seg_mask, gt_label, half_h, half_w, center_h, center_w, valid_mask_flag in zip(gt_masks, gt_labels, half_hs, half_ws, center_hs, center_ws, valid_mask_flags):
                if not valid_mask_flag:
                    continue
                upsampled_size = (featmap_sizes[0][0] * 4, featmap_sizes[0][1] * 4)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h-1)
                down = min(down_box, coord_h+1)
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)

                # squared
                cate_label[top:(down+1), left:(right+1)] = gt_label
                # ins
                seg_mask = F.interpolate(seg_mask.unsqueeze(0).unsqueeze(0), scale_factor=1. / output_stride).squeeze(0).squeeze(0)
                seg_mask = seg_mask.to(device=device)
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        label = int(i * num_grid + j)
                        ins_label[label, :seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_ind_label[label] = True

            # append to the output list
            ins_label_list.append(ins_label)
            ins_ind_label_list.append(ins_ind_label)
            cate_label_list.append(cate_label)

        # check flag
        assert ins_label_list[1].shape == (1296,200,272)
        assert ins_ind_label_list[1].shape == (1296,)
        assert cate_label_list[1].shape == (36, 36)
        return ins_label_list, ins_ind_label_list, cate_label_list
    
 
    # This function receive pred list from forward and post-process
    # Input:
        # ins_pred_list: list, len(fpn), (bz,S^2,Ori_H/4, Ori_W/4)
        # cate_pred_list: list, len(fpn), (bz,S,S,C-1)
        # ori_size: [ori_H, ori_W]
    # Output:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)

    def PostProcess(self,
                    ins_pred_list,
                    cate_pred_list,
                    ori_size=None):

        ## TODO: finish PostProcess
        NMS_sorted_scores_list = []
        NMS_sorted_cate_label_list = []
        NMS_sorted_ins_list = []

        bz = ins_pred_list[0].shape[0]
        N_fpn = len(ins_pred_list)
        for img_i in range(bz):
            # (all_level_S^2, ori_H/4, ori_W/4)
            ins_pred_img = torch.cat([ins_pred_list[i][img_i] for i in range(N_fpn)], dim=0)

            tmp_list = []
            # cate_pred: (bz,S,S,C-1) / after point_NMS
            # ins_pred: (bz, S^2, Ori_H/4, Ori_W/4) / after upsampling
            for fpn_i in range(N_fpn):
                cate_pred = cate_pred_list[fpn_i][img_i]        # (S,S,C-1)
                S_1, S_2, C = cate_pred.shape
                tmp_x = cate_pred.view(S_1 * S_2, C)       # (S_1 * S_2, C-1)
                tmp_list.append(tmp_x)
            # (all_level_S^2, C-1)
            cate_pred_img = torch.cat(tmp_list, dim=0)
            
            NMS_sorted_scores, NMS_sorted_cate_label, NMS_sorted_ins = self.PostProcessImg(ins_pred_img, cate_pred_img, ori_size)
            
            NMS_sorted_scores_list.append(NMS_sorted_scores)
            NMS_sorted_cate_label_list.append(NMS_sorted_cate_label)
            NMS_sorted_ins_list.append(NMS_sorted_ins)
        
        return NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list
    
    # This function Postprocess on single img
    # Input:
        # ins_pred_img: (all_level_S^2, ori_H/4, ori_W/4)
        # cate_pred_img: (all_level_S^2, C-1)
    # Output:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
    def PostProcessImg(self,
                       ins_pred_img,
                       cate_pred_img,
                       ori_size=None):

        ## TODO: PostProcess on single image.
        ins_thresh = self.postprocess_cfg['ins_thresh']
        cate_thresh = self.postprocess_cfg['cate_thresh']
        pre_NMS_num = self.postprocess_cfg['pre_NMS_num']
        keep_instance = self.postprocess_cfg['keep_instance']
        iou_thresh = self.postprocess_cfg['IoU_thresh']

        c_score, cate_pred  = torch.max(cate_pred_img, dim=1)

        score_thresh = c_score > cate_thresh
        c_score = c_score * score_thresh

        mask_thresh = ins_pred_img > ins_thresh # (all_level_S^2, ori_H/4, ori_W/4)

        score = c_score * torch.sum(ins_pred_img * mask_thresh, dim=(1,2)) / (torch.sum(mask_thresh, dim=(1,2)) + 1e-6)# (all_level_S^2,)

        _, sorted_indice = torch.sort(score, descending=True)
        sorted_indice = sorted_indice[:pre_NMS_num]

        sorted_score = score[sorted_indice]       
        sorted_cate = cate_pred[sorted_indice]
        sorted_mask = mask_thresh[sorted_indice] # binary mask
        sorted_ins = ins_pred_img[sorted_indice]

        decay_scores, ious = self.MatrixNMS(sorted_mask, sorted_score)

        NMS_sorted_scores, NMS_sorted_indice = torch.sort(decay_scores, descending=True)
        NMS_sorted_scores = sorted_score[NMS_sorted_indice] # (keep_instance,)
        NMS_sorted_cate_label = sorted_cate[NMS_sorted_indice] + 1 # exclude background
        NMS_sorted_ins = sorted_ins[NMS_sorted_indice]

        # threshold on ious
        for i in range(len(NMS_sorted_scores)):
            for j in range(i+1, len(NMS_sorted_scores)):
                if ious[i,j] > iou_thresh:
                    NMS_sorted_scores[j] = 0
                    NMS_sorted_cate_label[j] = 0
                    NMS_sorted_ins[j] = torch.zeros_like(NMS_sorted_ins[j])
        
        # resort
        # decay_scores, _ = self.MatrixNMS(NMS_sorted_ins, NMS_sorted_scores)
        NMS_sorted_scores, NMS_sorted_indice = torch.sort(NMS_sorted_scores, descending=True)
        # NMS_sorted_scores, NMS_sorted_indice = torch.sort(NMS_sorted_scores, descending=True)
        NMS_sorted_indice = NMS_sorted_indice[:keep_instance]

        NMS_sorted_scores = NMS_sorted_scores[NMS_sorted_indice] # (keep_instance,)
        NMS_sorted_cate_label = NMS_sorted_cate_label[NMS_sorted_indice] # exclude background
        NMS_sorted_ins = NMS_sorted_ins[NMS_sorted_indice]

        # resize to H_ori, W_ori
        NMS_sorted_ins = F.interpolate(NMS_sorted_ins.unsqueeze(0), scale_factor=(4, 4)).squeeze(0)

        NMS_sorted_ins = (NMS_sorted_ins > ins_thresh).float() # (keep_instance, ori_H, ori_W)

        return NMS_sorted_scores, NMS_sorted_cate_label, NMS_sorted_ins


    # This function perform matrix NMS
    # Input:
        # sorted_ins: (n_act, ori_H/4, ori_W/4)
        # sorted_scores: (n_act,)
    # Output:
        # decay_scores: (n_act,)
    def MatrixNMS(self, sorted_ins, sorted_scores, method='gauss', gauss_sigma=0.5):
        ## TODO: finish MatrixNMS
        n = len(sorted_scores)
        sorted_masks = sorted_ins.reshape(n, -1)

        if sorted_masks.dtype == torch.bool:
            sorted_masks = sorted_masks.to(torch.float32)
        intersection = torch.mm(sorted_masks, sorted_masks.T)
        areas = sorted_masks.sum(dim=1).expand(n, n)
        union = areas + areas.T - intersection + 1e-6
        ious = (intersection / union).triu(diagonal=1)

        ious_cmin = ious.min(0)[0].expand(n, n).T
        if method == 'gauss':
            decay = torch.exp(-(ious ** 2 - ious_cmin ** 2) / gauss_sigma)
        else:
            decay = (ious) / (ious_cmin)

        decay, _  = decay.min(dim=0)
        return sorted_scores * decay, ious

    # -----------------------------------
    ## The following code is for visualization
    # -----------------------------------
    # this function visualize the ground truth tensor
    # Input:
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
        # color_list: list, len(C-1)
        # img: (bz,3,Ori_H, Ori_W)
        ## self.strides: [8,8,16,32,32]
    def PlotGT(self,
               ins_gts_list,
               ins_ind_gts_list,
               cate_gts_list,
               color_list,
               img,
               iter):
        ## TODO: target image recover, for each image, recover their segmentation in 5 FPN levels.
        ## This is an important visual check flag.
        num_pyramids = len(ins_gts_list[0])

        for i in range(len(ins_gts_list)):
            for j in range(num_pyramids):
                fig, ax = plt.subplots(1)
                ax.imshow(img[i,:,:,:].cpu().numpy().transpose(1,2,0))

                output_dir = "../plot_gt/"
                os.makedirs(output_dir, exist_ok=True)

                if sum(ins_ind_gts_list[i][j]) == 0 :
                    # plt.savefig("../batch_" + str(iter) + "_image_" + str(i) + "_pyramid_level_" + str(j) + ".png")
                    filename = f"batch_{iter}_image_{i}_pyramid_level_{j}.png"
                    full_path = os.path.join(output_dir, filename)
                    
                    plt.savefig(full_path)
                    plt.show()
                    continue
                
                index = ins_ind_gts_list[i][j] > 0
                label = torch.flatten(cate_gts_list[i][j])[index]
                mask = ins_gts_list[i][j][index,:,:]
                mask = torch.unsqueeze(mask,1)

                reshaped_mask = F.interpolate(mask,(img.shape[2],img.shape[3]))
                # plot the mask
                reshaped_mask = reshaped_mask.squeeze(1).cpu().numpy()
                reshaped_mask = np.ma.masked_where(reshaped_mask == 0, reshaped_mask)
                for k in range(len(label)):
                    ax.imshow(reshaped_mask[k,:,:], cmap=color_list[label[k]-1], alpha=0.5, interpolation='none')
            
                filename = f"batch_{iter}_image_{i}_pyramid_level_{j}.png"
                full_path = os.path.join(output_dir, filename)
                    
                plt.savefig(full_path)
                plt.show()

    # This function plot the inference segmentation in img
    # Input:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
        # color_list: ["jet", "ocean", "Spectral"]
        # img: (bz, 3, ori_H, ori_W)
    def PlotInfer(self,
                  NMS_sorted_scores_list,
                  NMS_sorted_cate_label_list,
                  NMS_sorted_ins_list,
                  color_list,
                  img,
                  iter_ind):
        ## TODO: Plot predictions
        for i in range(len(img)):
            fig, ax = plt.subplots(1)
            ax.imshow(img[i,:,:,:].cpu().numpy().transpose(1,2,0))

            output_dir = "../plot_infer/"
            os.makedirs(output_dir, exist_ok=True)

            label = NMS_sorted_cate_label_list[i]
            mask = NMS_sorted_ins_list[i]

            mask = mask.cpu().numpy()
            mask = np.ma.masked_where(mask == 0, mask)
            for k in range(len(label)):
                ax.imshow(mask[k,:,:], cmap=color_list[label[k]-1], alpha=0.5, interpolation='none')
        
            filename = f"batch_{iter_ind}_image_{i}.png"
            full_path = os.path.join(output_dir, filename)
                
            plt.savefig(full_path)
            plt.show()

from backbone import *
if __name__ == '__main__':
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

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()


    resnet50_fpn = Resnet50Backbone()
    solo_head = SOLOHead(num_classes=4) ## class number is 4, because consider the background as one category.
    # loop the image
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(test_loader, 0):
        img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
        # fpn is a dict
        backout = resnet50_fpn(img)
        fpn_feat_list = list(backout.values())
        # make the target

        ## demo
        cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False)
        ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                         bbox_list,
                                                                         label_list,
                                                                         mask_list)
        mask_color_list = ["jet", "ocean", "Spectral"]
        solo_head.PlotGT(ins_gts_list,ins_ind_gts_list,cate_gts_list,mask_color_list,img,iter)
        
        if iter == 20:
            break


