import os
from os import path
import time
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from model.eval_network import STCN
from dataset.davis_test_dataset import DAVISTestDataset
from inference_core import InferenceCore
# from dataset.OSS import torch.load, save_pilimg

from progressbar import progressbar


"""
python eval_generic.py --model /gdata/limx/VOS/SAM/single-head-3frames-nonlocal3d-r50nonbais-dmy/saves/Aug09_09.03.46_s2-kl10-4gpu-75000-kl10-lr28-500000-kl10-lr-2e-5/model_75000.pth \
     --output predition/long-video --amp --mode two-frames-compress 
"""
parser = ArgumentParser()
parser.add_argument('--model', default='saves/stcn.pth', required=True)
parser.add_argument('--long_video_path', default='/gdata/limx/VOS/dataset/long-videos-rank')
parser.add_argument('--output')
parser.add_argument('--top', type=int, default=40)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--mem_every', default=3, type=int)
parser.add_argument('--mode', type=str, help="stm, two-frames, gt, last, compress, gt-compress, last-compress, two-frames-compress")
args = parser.parse_args()

long_video_path = args.long_video_path
out_path = args.output

# Simple setup
os.makedirs(out_path, exist_ok=True)

torch.autograd.set_grad_enabled(False)

# Setup Dataset, a small hack to use the image set in the 2017 folder because the 2016 one is of a different format
test_dataset = DAVISTestDataset(long_video_path+'/JPEGImages', imset='val.txt', single_object=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

# Load our checkpoint
prop_saved = torch.load(args.model)
top_k = args.top
prop_model = STCN().cuda().eval()
prop_model.load_state_dict(prop_saved)

total_process_time = 0
total_frames = 0

# Start eval
# for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):
for data in test_loader:
    with torch.cuda.amp.autocast(enabled=args.amp):
        rgb = data['rgb'].cuda()
        msk = data['gt'][0].cuda()
        info = data['info']
        name = info['name'][0]
        k = len(info['labels'][0])

        torch.cuda.synchronize()
        process_begin = time.time()

        processor = InferenceCore(args, prop_model, rgb, k, top_k=top_k, mem_every=args.mem_every)
        processor.interact(msk[:,0], 0, rgb.shape[1])

        # Do unpad -> upsample to original size 
        out_masks = torch.zeros((processor.t, 1, *rgb.shape[-2:]), dtype=torch.float32, device='cuda')
        for ti in range(processor.t):
            prob = processor.prob[:,ti]

            if processor.pad[2]+processor.pad[3] > 0:
                prob = prob[:,:,processor.pad[2]:-processor.pad[3],:]
            if processor.pad[0]+processor.pad[1] > 0:
                prob = prob[:,:,:,processor.pad[0]:-processor.pad[1]]

            out_masks[ti] = torch.argmax(prob, dim=0)*255
        
        out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

        torch.cuda.synchronize()
        total_process_time += time.time() - process_begin
        total_frames += out_masks.shape[0]

        this_out_path = path.join(out_path, name)
        os.makedirs(this_out_path, exist_ok=True)
        for f in range(out_masks.shape[0]):
            img_E = Image.fromarray(out_masks[f])
            img_E.save(os.path.join(this_out_path, '{:05d}.png'.format(f)))
            # save_pilimg(img_E, os.path.join(this_out_path, '{:05d}.png'.format(f)), ext='png')
            print(this_out_path)
            

        del rgb
        del msk
        del processor

print('Total processing time: ', total_process_time)
print('Total processed frames: ', total_frames)
print('FPS: ', total_frames / total_process_time)