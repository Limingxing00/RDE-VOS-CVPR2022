"""
YouTubeVOS has a label structure that is more complicated than DAVIS 
Labels might not appear on the first frame (there might be no labels at all in the first frame)
Labels might not even appear on the same frame (i.e. Object 0 at frame 10, and object 1 at frame 15)
0 does not mean background -- it is simply "no-label"
and object indices might not be in order, there are missing indices somewhere in the validation set

Dealing with these makes the logic a bit convoluted here
It is not necessarily hacky but do understand that it is not as straightforward as DAVIS

Validation/test set.
"""


import os
from os import path
from argparse import ArgumentParser
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from model.eval_network import RDE_VOS
from dataset.yv_test_dataset import YouTubeVOSTestDataset
from util.tensor_util import unpad
from inference_core_yv import InferenceCore
# from dataset.OSS import Image.open
# from dataset.OSS import torch.load, read_json, save_pilimg
from progressbar import progressbar

"""
python eval_youtube.py --model /gdata/limx/VOS/SAM/single-head-3frames-nonlocal3d-r50nonbais-dmy/saves/Aug09_09.03.46_s2-kl10-4gpu-75000-kl10-lr28-500000-kl10-lr-2e-5/model_75000.pth \
 --output_all \
 --output prediction/dmy/Aug09_09.03.46_s2-kl10-4gpu-75000-kl10-lr28-500000-kl10-lr-2e-5/youtube-every-20-top-20 --amp --mode two-frames-compress --mem_every 3 --top 20
"""
parser = ArgumentParser()
parser.add_argument('--model', default='mingxing/code/RDE_VOS/Jul23_17.07.35_s3-perturb-group-v3-kl10-distillation-strong-v2-df2-10-df4-10/model_60000.pth')
parser.add_argument('--yv_path', default='/gdata/limx/VOS/dataset/YouTube-VOS')

parser.add_argument('--output_all', help=
"""
We will output all the frames if this is set to true.
Otherwise only a subset will be outputted, as determined by meta.json to save disk space.
For ensemble, all the sources must have this setting unified.
""", action='store_true')

parser.add_argument('--output')
parser.add_argument('--split', help='valid/test', default='valid')
parser.add_argument('--top', type=int, default=20)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--mem_every', default=3, type=int)
# parser.add_argument('--include_last', help='include last frame as temporary memory?', action='store_true')
parser.add_argument('--mode', type=str, help="stm, two-frames, gt, last, compress, gt-compress, last-compress, two-frames-compress")

args = parser.parse_args()

yv_path = args.yv_path
out_path = args.output

if __name__=="__main__":
    # Simple setup
    os.makedirs(out_path, exist_ok=True)
    palette = Image.open(path.expanduser(yv_path + '/valid/Annotations/0a49f5265b/00000.png')).getpalette()

    torch.autograd.set_grad_enabled(False)

    # Load the json if we have to
    if not args.output_all:
        with open(path.join(yv_path, args.split, 'meta.json')) as f:
            meta = json.load(f)['videos']


    # Setup Dataset
    test_dataset = YouTubeVOSTestDataset(data_root=yv_path, split=args.split)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Load our checkpoint
    prop_saved = torch.load(args.model)
    top_k = args.top
    prop_model = RDE_VOS().cuda().eval()
    prop_model.load_state_dict(prop_saved)

    # Start eval
    for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):
    # for data in test_loader:
        with torch.cuda.amp.autocast(enabled=args.amp):
            rgb = data['rgb']
            msk = data['gt'][0]
            info = data['info']
            name = info['name'][0]
            num_objects = len(info['labels'][0])
            gt_obj = info['gt_obj']
            size = info['size']

            # Load the required set of frames (if we don't need all)
            req_frames = None
            if not args.output_all:
                req_frames = []
                objects = meta[name]['objects']
                for key, value in objects.items():
                    req_frames.extend(value['frames'])

                # Map the frame names to indices
                req_frames_names = set(req_frames)
                req_frames = []
                for fi in range(rgb.shape[1]):
                    frame_name = info['frames'][fi][0][:-4]
                    if frame_name in req_frames_names:
                        req_frames.append(fi)
                req_frames = sorted(req_frames)

            # Frames with labels, but they are not exhaustively labeled
            frames_with_gt = sorted(list(gt_obj.keys()))

            processor = InferenceCore(args, prop_model, rgb, num_objects=num_objects, top_k=top_k, 
                                        mem_every=args.mem_every, mode=args.mode, 
                                        req_frames=req_frames)
            # min_idx tells us the starting point of propagation
            # Propagating before there are labels is not useful
            min_idx = 99999
            for i, frame_idx in enumerate(frames_with_gt):
                min_idx = min(frame_idx, min_idx)
                # Note that there might be more than one label per frame
                obj_idx = gt_obj[frame_idx][0].tolist()
                # Map the possibly non-continuous labels into a continuous scheme
                obj_idx = [info['label_convert'][o].item() for o in obj_idx]

                # Append the background label
                with_bg_msk = torch.cat([
                    1 - torch.sum(msk[:,frame_idx], dim=0, keepdim=True),
                    msk[:,frame_idx],
                ], 0).cuda()

                # We perform propagation from the current frame to the next frame with label
                if i == len(frames_with_gt) - 1:
                    processor.interact(with_bg_msk, frame_idx, rgb.shape[1], obj_idx)
                else:
                    processor.interact(with_bg_msk, frame_idx, frames_with_gt[i+1]+1, obj_idx)

            # Do unpad -> upsample to original size (we made it 480p)
            out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')
            for ti in range(processor.t):
                prob = unpad(processor.prob[:,ti], processor.pad)
                prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
                out_masks[ti] = torch.argmax(prob, dim=0)

            out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

            # Remap the indices to the original domain
            idx_masks = np.zeros_like(out_masks)
            for i in range(1, num_objects+1):
                backward_idx = info['label_backward'][i].item()
                idx_masks[out_masks==i] = backward_idx
            
            # Save the results
            this_out_path = path.join(out_path, 'Annotations', name)
            os.makedirs(this_out_path, exist_ok=True)
            print("save_path", this_out_path)
            for f in range(idx_masks.shape[0]):
                if f >= min_idx:
                    if args.output_all or (f in req_frames):
                        # print(os.path.join(this_out_path, info['frames'][f][0].replace('.jpg','.png')))
                        img_E = Image.fromarray(idx_masks[f])
                        img_E.putpalette(palette)
                        img_E.save(os.path.join(this_out_path, info['frames'][f][0].replace('.jpg','.png')))
                        # save_pilimg(img_E, os.path.join(this_out_path, info['frames'][f][0].replace('.jpg','.png')), ext='png')
                        

            del rgb
            del msk
            del processor
