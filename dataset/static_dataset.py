import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from dataset.range_transform import im_normalization, im_mean
from dataset.tps import random_tps_warp
from dataset.reseed import reseed
# from dataset.OSS import Image.open, os.listdir


def get_random_structure(size):
    choice = np.random.randint(1, 5)

    if choice == 1:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif choice == 2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif choice == 3:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size//2))
    elif choice == 4:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size//2, size))

def random_dilate(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.dilate(seg,kernel,iterations = 1)
    return seg

def random_erode(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.erode(seg,kernel,iterations = 1)
    return seg

def compute_iou(seg, gt):
    intersection = seg*gt
    union = seg+gt
    return (np.count_nonzero(intersection) + 1e-6) / (np.count_nonzero(union) + 1e-6)

def perturb_mask(gt, iou_target=0.9):
    if iou_target==1:
        return gt
    h, w = gt.shape
    seg = gt.copy()

    # _, seg = cv2.threshold(seg, 127, 255, 0)

    # Rare case
    if h <= 2 or w <= 2:
        print('GT too small, returning original')
        return seg

    # Do a bunch of random operations
    for _ in range(250):
        for _ in range(4):
            lx, ly = np.random.randint(w), np.random.randint(h)
            lw, lh = np.random.randint(lx+1,w+1), np.random.randint(ly+1,h+1)

            # Randomly set one pixel to 1/0. With the following dilate/erode, we can create holes/external regions
            if np.random.rand() < 0.1:
                cx = int((lx + lw) / 2)
                cy = int((ly + lh) / 2)
                seg[cy, cx] = np.random.randint(2) * 255

            # Dilate/erode
            if np.random.rand() < 0.5:
                seg[ly:lh, lx:lw] = random_dilate(seg[ly:lh, lx:lw])
            else:
                seg[ly:lh, lx:lw] = random_erode(seg[ly:lh, lx:lw])

        if compute_iou(seg, gt) < iou_target:
            break

    return seg

class StaticTransformDataset(Dataset):
    """
    Generate pseudo VOS data by applying random transforms on static images.
    Single-object only.

    Method 0 - FSS style (class/1.jpg class/1.png)
    Method 1 - Others style (XXX.jpg XXX.png)
    """
    def __init__(self, para, root, method=0):
        self.root = root
        self.method = method
        self.para = para

        if method == 0:
            # Get images
            self.im_list = []
            classes = os.listdir(self.root)
            for c in classes:
                imgs = os.listdir(path.join(root, c))
                jpg_list = [im for im in imgs if 'jpg' in im[-3:].lower()]

                joint_list = [path.join(root, c, im) for im in jpg_list]
                self.im_list.extend(joint_list)

        elif method == 1:
            self.im_list = [path.join(self.root, im) for im in os.listdir(self.root) if '.jpg' in im]

        print('%d images found in %s' % (len(self.im_list), root))

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.05, 0.05, 0), # No hue change here as that's not realistic
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.9,1.1), shear=10, resample=Image.BICUBIC, fillcolor=im_mean),
            transforms.Resize(384, Image.BICUBIC),
            transforms.RandomCrop((384, 384), pad_if_needed=True, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.9,1.1), shear=10, resample=Image.BICUBIC, fillcolor=0),
            transforms.Resize(384, Image.NEAREST),
            transforms.RandomCrop((384, 384), pad_if_needed=True, fill=0),
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.05, 0.05, 0.05),
            transforms.RandomGrayscale(0.05),
        ])

        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.5), fillcolor=im_mean),
            transforms.RandomHorizontalFlip(),
        ])

        self.all_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.5), fillcolor=0),
            transforms.RandomHorizontalFlip(),
        ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

        self.final_gt_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def __getitem__(self, idx):
        iou_max = self.para["perturb_max"]
        iou_min = self.para["perturb_min"]
        
        iou_target = np.random.rand()*(iou_max-iou_min) + iou_min
        im = Image.open(self.im_list[idx]).convert('RGB')

        if self.method == 0:
            gt = Image.open(self.im_list[idx][:-3]+'png').convert('L')
        else:
            gt = Image.open(self.im_list[idx].replace('.jpg','.png')).convert('L')

        sequence_seed = np.random.randint(2147483647)

        images = []
        masks = []
        for _ in range(3):
            reseed(sequence_seed)
            this_im = self.all_im_dual_transform(im)
            this_im = self.all_im_lone_transform(this_im)
            reseed(sequence_seed)
            this_gt = self.all_gt_dual_transform(gt)

            pairwise_seed = np.random.randint(2147483647)
            reseed(pairwise_seed)
            this_im = self.pair_im_dual_transform(this_im)
            this_im = self.pair_im_lone_transform(this_im)
            reseed(pairwise_seed)
            this_gt = self.pair_gt_dual_transform(this_gt)

            # Use TPS only some of the times
            # Not because TPS is bad -- just that it is too slow and I need to speed up data loading
            if np.random.rand() < 0.33:
                this_im, this_gt = random_tps_warp(this_im, this_gt, scale=0.02)

            this_im = self.final_im_transform(this_im)
            this_gt = self.final_gt_transform(this_gt)

            images.append(this_im)
            masks.append(this_gt)

        images = torch.stack(images, 0)
        masks = torch.stack(masks, 0)

        info = {}
        info['name'] = self.im_list[idx]

        cls_gt = np.zeros((3, 384, 384), dtype=np.int)

        if masks.max()>1:
            masks[masks>1] = 1
        pmask = masks.clone()
        cls_gt[masks[:,0] > 0.5] = 1

        if masks[0,0].max()==1:
            pmask[0, 0] = torch.from_numpy(perturb_mask(255*pmask[0, 0].numpy(), iou_target)/255)


        data = {
            'rgb': images,
            'gt': masks,
            'pgt': pmask,
            'cls_gt': cls_gt,
            'info': info
        }

        return data


    def __len__(self):
        return len(self.im_list)
