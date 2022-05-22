import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
# from torchvision.transforms import Image
from PIL import Image
import numpy as np
import cv2

from dataset.range_transform import im_normalization, im_mean
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


class VOSDataset(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, para, im_root, gt_root, max_jump, is_bl, subset=None):
        self.im_root = im_root
        self.gt_root = gt_root
        self.max_jump = max_jump
        self.is_bl = is_bl
        self.para = para

        self.videos = []
        self.frames = {}

        vid_list = sorted(os.listdir(self.im_root))
        # Pre-filtering
        for vid in vid_list:
            if subset is not None:
                if vid not in subset:
                    continue
            frames = sorted(os.listdir(os.path.join(self.im_root, vid)))
            if len(frames) < 3:
                continue
            self.frames[vid] = frames
            self.videos.append(vid)

        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), im_root))

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.01, 0.01, 0.01, 0),
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, resample=Image.BICUBIC, fillcolor=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, resample=Image.NEAREST, fillcolor=0),
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])

        if self.is_bl:
            # Use a different cropping scheme for the blender dataset because the image size is different
            self.all_im_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.25, 1.00), interpolation=Image.BICUBIC)
            ])

            self.all_gt_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.25, 1.00), interpolation=Image.NEAREST)
            ])
        else:
            self.all_im_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=Image.BICUBIC)
            ])

            self.all_gt_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=Image.NEAREST)
            ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

    def generate_perturbation(self, mask, iou_target):
        """
        mask: (3, 1, 384, 384) T, 1, H, W
        iou_target: N
        return: N
        """
        temp = []
        for i in range(len(iou_target)):
            temp.append(perturb_mask(255*mask[0, 0], iou_target[i])/255)
        
        return np.stack(temp, axis=0)[:,np.newaxis,...]


    from retrying import retry
    @retry(stop_max_attempt_number=3)
    def __getitem__(self, idx):
        video = self.videos[idx]
        frames = self.frames[video]
        if len(frames)<12:
            while(len(frames)>12):
                idx = max(0, idx - np.random.randint(10, 20))
                video = self.videos[idx]
                frames = self.frames[video]
                print("skip...")

        info = {}
        info['name'] = video
        iou_max = self.para["perturb_max"]
        iou_min = self.para["perturb_min"]
        group = self.para["perturb_group"]

        iou_target = []
        for _ in range(group):
            iou_target.append(np.random.rand()*(iou_max-iou_min) + iou_min)
        # print("iou_target", iou_target)

        vid_im_path = path.join(self.im_root, video)
        vid_gt_path = path.join(self.gt_root, video)
        
            

        trials = 0
        frames_num = 5
        while trials < 5:
            info['frames'] = [] # Appended with actual frames
            
            # Don't want to bias towards beginning/end
            this_max_jump = min(len(frames), self.max_jump) # len(frames) 36 self.max_jump 1

            start_idx = np.random.randint(max(self.max_jump//2+1, len(frames)-frames_num*this_max_jump+1))
            f1_idx = start_idx + np.random.randint(this_max_jump+1) + 1 # not include this_max_jump+1
            f1_idx = min(f1_idx, len(frames)-this_max_jump, len(frames)-1)

            f2_idx = f1_idx + np.random.randint(this_max_jump+1) + 1
            f2_idx = min(f2_idx, len(frames)-this_max_jump//2, len(frames)-1)

            # my setting 
            f3_idx = f2_idx + np.random.randint(this_max_jump+1) + 1
            f3_idx = min(f3_idx, len(frames)-this_max_jump//3, len(frames)-1)

            f4_idx = f3_idx + np.random.randint(this_max_jump+1) + 1
            f4_idx = min(f4_idx, len(frames)-this_max_jump//4, len(frames)-1)

            frames_idx = [start_idx, f1_idx, f2_idx, f3_idx, f4_idx]
            # if len(set(frames_idx))!=len(frames_idx):
            #     print("bad", frames_idx)
            #     print(vid_im_path, len(frames), self.max_jump)
            #     # continue
            # print(frames_idx)
            if np.random.rand() < 0.5:
                # Reverse time
                frames_idx = frames_idx[::-1]

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            target_object = None
            for f_idx in frames_idx:
                jpg_name = frames[f_idx][:-4] + '.jpg'
                png_name = frames[f_idx][:-4] + '.png'
                info['frames'].append(jpg_name)

                reseed(sequence_seed)
                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)
                reseed(sequence_seed)
                this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')
                this_gt = self.all_gt_dual_transform(this_gt)

                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im)
                this_im = self.pair_im_lone_transform(this_im)
                reseed(pairwise_seed)
                this_gt = self.pair_gt_dual_transform(this_gt)

                this_im = self.final_im_transform(this_im)
                this_gt = np.array(this_gt)

                images.append(this_im)
                masks.append(this_gt)

            images = torch.stack(images, 0)

            labels = np.unique(masks[0])
            # Remove background
            labels = labels[labels!=0]

            if self.is_bl:
                # Find large enough labels
                good_lables = []
                for l in labels:
                    pixel_sum = (masks[0]==l).sum()
                    if pixel_sum > 10*10:
                        # OK if the object is always this small
                        # Not OK if it is actually much bigger
                        if pixel_sum > 30*30:
                            good_lables.append(l)
                        elif max((masks[1]==l).sum(), (masks[2]==l).sum()) < 20*20:
                            good_lables.append(l)
                labels = np.array(good_lables, dtype=np.uint8)
            
            if len(labels) == 0:
                target_object = -1 # all black if no objects
                has_second_object = False
                trials += 1
            else:
                target_object = np.random.choice(labels)
                has_second_object = (len(labels) > 1)
                if has_second_object:
                    labels = labels[labels!=target_object]
                    second_object = np.random.choice(labels)
                break

        masks = np.stack(masks, 0)
        tar_masks = (masks==target_object).astype(np.float32)[:,np.newaxis,:,:]

        first_pmasks = self.generate_perturbation(mask=tar_masks, iou_target=iou_target)
        # tar_masks[0, 0] = first_pmasks[0] cancel the gt frame change
        if has_second_object:
            sec_masks = (masks==second_object).astype(np.float32)[:,np.newaxis,:,:]
            sec_pmasks = self.generate_perturbation(mask=sec_masks, iou_target=iou_target)
            # sec_masks[0, 0] = sec_pmasks[0]
            selector = torch.FloatTensor([1, 1])
            
        else:
            sec_masks = np.zeros_like(tar_masks)
            sec_pmasks = np.zeros_like(first_pmasks)
            selector = torch.FloatTensor([1, 0])

        cls_gt = np.zeros((frames_num, 384, 384), dtype=np.int)
        
        cls_gt[tar_masks[:,0] > 0.5] = 1
        cls_gt[sec_masks[:,0] > 0.5] = 2

        data = {
            'rgb': images,
            'gt': tar_masks,        # (3, 1, 384, 384) T, 1, H, W
            'sec_gt': sec_masks,
            'pgt': first_pmasks,
            'sec_pgt': sec_pmasks,
            'cls_gt': cls_gt,
            'selector': selector,
            'info': info,
        }


        # cv2.imwrite("perturbation1st.tiff", data['gt'][0, 0])
        # cv2.imwrite("perturbation1st_img.tiff", np.array(data['rgb'][0, 0]))
        return data

    def __len__(self):
        return len(self.videos)


