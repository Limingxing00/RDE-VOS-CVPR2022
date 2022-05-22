import os
from os import path
import time
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from model.eval_network import RDE_VOS
from dataset.davis_test_dataset import DAVISTestDataset
from util.tensor_util import unpad
from inference_core import InferenceCore
# from dataset.OSS import Image.open
# from dataset.OSS import load_model, save_pilimg
from progressbar import progressbar


"""
cd /gdata/limx/VOS/SAM/single-head-5frames-nonlocal3d-r50nonbais-dmy &&\
CUDA_VISIBLE_DEVICES=2 python eval_davis.py \
--model /gdata/limx/VOS/SAM/single-head-3frames-nonlocal3d-r50nonbais-dmy/saves/Aug09_09.03.46_s2-kl10-4gpu-75000-kl10-lr28-500000-kl10-lr-2e-5/model_75000.pth \
--output prediction/dmy/debug \
--repeat 0 \
--mode two-frames-compress --amp  --mem_every 3 --top 40
"""
parser = ArgumentParser()
parser.add_argument('--model', default='..', required=True)
parser.add_argument('--davis_path', default='/gdata/limx/VOS/dataset/DAVIS-2017')
parser.add_argument('--output')
parser.add_argument('--split', help='val/testdev', default='val')
parser.add_argument('--top', type=int, default=40)
parser.add_argument('--repeat', type=int, default=0)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--mem_every', default=3, type=int)
parser.add_argument('--include_last', help='include last frame as temporary memory?', action='store_true')
# parser.add_argument('--compress', action='store_true')
parser.add_argument('--mode', type=str, help="stm, two-frames, gt, last, compress, gt-compress, last-compress, two-frames-compress")
args = parser.parse_args()

davis_path = args.davis_path
out_path = args.output


if __name__=="__main__":
    # Simple setup
    os.makedirs(out_path, exist_ok=True)
    palette = Image.open(path.expanduser(davis_path + '/trainval/Annotations/480p/blackswan/00000.png')).getpalette()

    torch.autograd.set_grad_enabled(False)

    # Setup Dataset
    if args.split == 'val':
        test_dataset = DAVISTestDataset(davis_path+'/trainval', imset='2017/val.txt')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    elif args.split == 'testdev':
        test_dataset = DAVISTestDataset(davis_path+'/test-dev', imset='2017/test-dev.txt', test_dev=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    else:
        raise NotImplementedError

    # Load our checkpoint
    prop_saved = torch.load(args.model)
    top_k = args.top
    prop_model = RDE_VOS(args.repeat).cuda().eval()
    prop_model.load_state_dict(prop_saved, strict=False)

    total_process_time = 0
    total_frames = 0

    # Start eval
    for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):

        with torch.cuda.amp.autocast(enabled=args.amp):
            rgb = data['rgb'].cuda()
            msk = data['gt'][0].cuda()
            info = data['info']
            name = info['name'][0]
            k = len(info['labels'][0])
            size = info['size_480p']

            torch.cuda.synchronize()
            process_begin = time.time()

            processor = InferenceCore(args, prop_model, rgb, k, top_k=top_k, 
                            mem_every=args.mem_every, include_last=args.include_last)
            processor.interact(msk[:,0], 0, rgb.shape[1])

            # Do unpad -> upsample to original size 
            out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')
            for ti in range(processor.t):
                prob = unpad(processor.prob[:,ti], processor.pad)
                prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
                out_masks[ti] = torch.argmax(prob, dim=0)

            torch.cuda.synchronize()
            total_process_time += time.time() - process_begin

            out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)
            total_frames += out_masks.shape[0]

            # Save the results
            this_out_path = path.join(out_path, name)
            os.makedirs(this_out_path, exist_ok=True)
            for f in range(out_masks.shape[0]):
                img_E = Image.fromarray(out_masks[f])
                img_E.putpalette(palette)
                img_E.save(os.path.join(this_out_path, '{:05d}.png'.format(f)))
                # save_pilimg(img_E, os.path.join(this_out_path, '{:05d}.png'.format(f)), ext='png')

            del rgb
            del msk
            del processor

    print('Total processing time: ', total_process_time)
    print('Total processed frames: ', total_frames)
    print('FPS: ', total_frames / total_process_time)