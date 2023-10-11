## Author: Lishuo Pan 2020/4/18
import os
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        # TODO: load dataset, make mask list
        self.imgs_data = h5py.File(path[0],'r')['data']
        self.mask = h5py.File(path[1],'r')['data']
        self.labels = np.load(path[2], allow_pickle=True)
        self.bbox = np.load(path[3], allow_pickle=True)

        self.normalize = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        
        n_objects_per_img = [len(self.labels[i]) for i in range(len(self.labels))]     
        self.mask_index = np.cumsum(n_objects_per_img)                                        
        self.mask_index = np.concatenate([np.array([0]), self.mask_index])

    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
    def __getitem__(self, index):
        # TODO: __getitem__
        # images
        img = self.imgs_data[index] # (3, 300, 400)

        # annotation
        # label: start counting from 1
        label = torch.tensor(self.labels[index], dtype=torch.long)

        # collect all object mask for the image
        mask_begin = self.mask_index[index]
        mask_list = []
        for i in range(len(label)):
            # get the mask of the ith object in the image
            mask_np = self.mask[mask_begin + i] * 1.0
            mask_tmp = torch.tensor(mask_np, dtype=torch.float)
            mask_list.append(mask_tmp)
        # (n_obj, 300, 400)
        mask = torch.stack(mask_list)

        # bbox
        bbox_np = self.bbox[index]
        bbox = torch.tensor(bbox_np, dtype=torch.float)

        # preprocess
        transed_img, transed_mask, transed_bbox = self.pre_process_batch(img, mask, bbox)

        # check flag
        assert transed_img.shape == (3, 800, 1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]
        return transed_img, label, transed_mask, transed_bbox
    
    def __len__(self):
        return len(self.imgs_data)

    # This function take care of the pre-process of img,mask,bbox
    # in the input mini-batch
    # input:
        # img: 3*300*400
        # mask: 3*300*400
        # bbox: n_box*4
    def pre_process_batch(self, img, mask, bbox):
        # TODO: image preprocess
        # Normalize pixel values to [0,1].
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float)

        # Rescale the image to 800×1066.
        img = F.interpolate(img.unsqueeze(0), size=(800, 1066)).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=(800, 1066)).squeeze(0)
        
        # Normalize each channel with means and standard deviations.
        img = self.normalize(img)

        # Zero pad the image to 800×1088: left and right each 11 pixels.
        img = F.pad(img, (11, 11, 0, 0), "constant", 0)
        mask = F.pad(mask, (11, 11, 0, 0), "constant", 0)

        # bounding box coordinates are rescaled accordingly.
        bbox[:,0] = bbox[:,0] * 1066 / 400 + 11
        bbox[:,1] = bbox[:,1] * 800 / 300
        bbox[:,2] = bbox[:,2] * 1066 / 400 + 11
        bbox[:,3] = bbox[:,3] * 800 / 300

        # check flag
        assert img.shape == (3, 800, 1088)
        assert bbox.shape[0] == mask.shape[0]
        return img, mask, bbox


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    # output:
        # img: (bz, 3, 800, 1088)
        # label_list: list, len:bz, each (n_obj,)
        # transed_mask_list: list, len:bz, each (n_obj, 800,1088)
        # transed_bbox_list: list, len:bz, each (n_obj, 4)
        # img: (bz, 3, 300, 400)
    def collect_fn(self, batch):
        # TODO: collect_fn
        img_list = []
        label_list = []
        transed_mask_list = []
        transed_bbox_list = []

        for transed_img, label, transed_mask, transed_bbox in batch:
            img_list.append(transed_img)
            label_list.append(label)
            transed_mask_list.append(transed_mask)
            transed_bbox_list.append(transed_bbox)
        return torch.stack(img_list, dim=0), label_list, transed_mask_list, transed_bbox_list

    def loader(self):
        # TODO: return a dataloader
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=self.collect_fn)

## Visualize debugging
if __name__ == '__main__':
    # file path and make a list
    imgs_path = '../data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = '../data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = '../data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = '../data/hw3_mycocodata_bboxes_comp_zlib.npy'
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
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    mask_color_list = ["jet", "ocean", "Spectral", "spring", "cool"]
    # loop the image
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(test_loader, 0):

        img, label, mask, bbox = [data[i] for i in range(len(data))]
        # check flag
        assert img.shape == (batch_size, 3, 800, 1088)
        assert len(mask) == batch_size

        label = [label_img.to(device) for label_img in label]
        mask = [mask_img.to(device) for mask_img in mask]
        bbox = [bbox_img.to(device) for bbox_img in bbox]

        # plot the origin img
        for i in range(batch_size):
            ## TODO: plot images with annotations
            # plot the img
            fig, ax = plt.subplots(1)
            image = img[i].permute(1,2,0).cpu().numpy()
            image = np.clip(image, 0, 1)
            ax.imshow(image)
            
            for j in range(len(label[i])):
                # plot the bbox
                x1, y1, x2, y2 = bbox[i][j]
                w = x2 - x1
                h = y2 - y1
                rect = patches.Rectangle((x1,y1),w,h,linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)

                # plot the mask
                mask_np = mask[i][j].cpu().numpy()
                mask_np = np.ma.masked_where(mask_np == 0, mask_np)
                ax.imshow(mask_np, cmap=mask_color_list[label[i][j]-1], alpha=0.5, interpolation='none')

            # Check if the "testfig" directory exists, and create it if not
            output_dir = "../testfig/"
            os.makedirs(output_dir, exist_ok=True)

            # Save the figure to the "testfig" directory
            plt.savefig(os.path.join(output_dir, f"visualtrainset{iter}_{i}.png"))
            plt.show()

        if iter == 20:
            break

