"""
model.py - warpper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""


import os
import time
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from model.network import RDE_VOS
from model.losses import LossComputer, iou_hooks_mo, iou_hooks_so
from util.log_integrator import Integrator
from util.image_saver import pool_pairs
from util.warm_up import WarmupMultiStepLR
# from dataset.OSS import load_model, snapshot, save_logger

class KLDivLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super(KLDivLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.KLDivLoss()

    def forward(self, s_logits, t_logits):
        # s_logits -> t_logits
        return self.criterion(
            F.log_softmax(s_logits / self.temperature, dim=1),
            F.softmax(t_logits / self.temperature, dim=1),
        ) * (self.temperature ** 2)

class RDE_VOSModel:
    def __init__(self, para, logger=None, save_path=None, local_rank=0, world_size=1, long_id=""):
        self.para = para
        self.single_object = para['single_object']
        self.local_rank = local_rank

        self.RDE_VOS = nn.parallel.DistributedDataParallel(
            RDE_VOS(self.single_object, repeat=para["repeat"], norm=para["norm"]
            ).cuda(), 
            device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

        # Setup logger when local_rank=0
        self.long_id = long_id
        self.logger = logger
        self.save_path = save_path
        if logger is not None:
            self.last_time = time.time()
        self.train_integrator = Integrator(self.logger, distributed=True, local_rank=local_rank, world_size=world_size)
        if self.single_object:
            self.train_integrator.add_hook(iou_hooks_so)
        else:
            self.train_integrator.add_hook(iou_hooks_mo)
        self.loss_computer = LossComputer(para)

        self.train()
        self.optimizer = optim.Adam(filter(
            lambda p: p.requires_grad, self.RDE_VOS.parameters()), lr=para['lr'], weight_decay=1e-7)
        if para['warm_up']:
            # maybe warm up
            warm_up_iter = int(para['iterations']/100)
            self.scheduler = WarmupMultiStepLR(
                optimizer=self.optimizer,
                milestones=para['steps'],
                gamma=para['gamma'],
                warmup_factor=1/warm_up_iter,
                warmup_iters=warm_up_iter,
                warmup_method='linear'
            )

            print("warm up setting...")
            print("step, {}, gamma, {}, warmup_factor, {}, warmup_iters, {}".format(
                para['steps'], para['gamma'], 1/warm_up_iter, warm_up_iter
            ))
        else:
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, para['steps'], para['gamma'])

        if para['amp']:
            self.scaler = torch.cuda.amp.GradScaler()

        # Logging info
        self.report_interval = 100
        self.save_im_interval = 200000
        self.save_model_interval = para['save_interval']
        if para['debug']:
            self.report_interval = self.save_im_interval = 1
        # self.mse = nn.MSELoss()
        self.klloss = KLDivLoss(temperature=para['temperature'])

    def uploadLogger(self): # TODO: check it
        # only rank=0(default) has long_id
        local_rank = torch.distributed.get_rank() 
        if local_rank==0:
            root = self.para['out_root']

            # print("check - oss model saves in:", os.path.join(root, "tensorboard", 'events.out.tfevents.%s' % self.para['id'], "%s"% long_id, '_checkpoint.pth'))
            file = os.listdir(os.path.join('.', 'log', '%s' % self.long_id))[0]
            save_logger(os.path.join(root, "tensorboard", self.para['id'], 'events.out.tfevents_%s'% (self.para['id'])), os.path.join('.', 'log', '%s' % self.long_id, file))

    def infer_model(self, it):
        if torch.distributed.get_rank()==0:
            model_path = self.save_path + ('_%s.pth' % it)
            out_dir = self.para['id']
            print("Start inference...")
            cmd = [
                "python", "eval_davis.py",
                "--output", 
                "mingxing/gdata1/RDE_VOS/"+out_dir+"/"+str(it),
                '--model',
                model_path,
                '--mode', 'two-frames-compress',
                '--mem_every', '3',
                '--amp',
            ]
            cmd = " ".join(cmd)
            print(cmd)
            subprocess.run(cmd, shell=True)
        if torch.distributed.get_rank() not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache
        if torch.distributed.get_rank() == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab



    def do_pass(self, data, it=0):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)
        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        out = {}
        Fs = data['rgb']
        Ms = data['gt']
        if not self.single_object:
            pgt1 = data['pgt']
            pgt2 = data['sec_pgt']


        with torch.cuda.amp.autocast(enabled=self.para['amp']):
            # key features never change, compute once
            k16, kf16_thin, kf16, kf8, kf4 = self.RDE_VOS('encode_key', Fs) # K16  [4, 64, 3, 24, 24]

            if self.single_object:
                pgt = data['pgt']
                ref_v = self.RDE_VOS('encode_value', Fs[:,0], kf16[:,0], pgt[:,0])
                # gt-branch
                v_2 = self.RDE_VOS('encode_value', Fs[:,0], kf16[:,0], Ms[:,0])


                # Segment frame 1 with frame 0
                prev_logits, prev_mask, _ = self.RDE_VOS('segment', 
                        k16[:,:,1], kf16_thin[:,1], kf8[:,1], kf4[:,1], 
                        k16[:,:,0:1], ref_v)
                prev_v = self.RDE_VOS('encode_value', Fs[:,1], kf16[:,1], prev_mask)

                values = torch.cat([ref_v, prev_v], 2)


                # Segment frame 2 with frame 0 and 1
                this_logits, this_mask, _  = self.RDE_VOS('segment', 
                        k16[:,:,2], kf16_thin[:,2], kf8[:,2], kf4[:,2], 
                        k16[:,:,0:2], values)

                out['mask_1'] = prev_mask
                out['mask_2'] = this_mask
                out['logits_1'] = prev_logits
                out['logits_2'] = this_logits
            else:
                sec_Ms = data['sec_gt'] # N, T, 1, H, W
                selector = data['selector']

                ref_v1 = self.RDE_VOS('encode_value', Fs[:,0], kf16[:,0], pgt1[:,0], pgt2[:,0])
                ref_v2 = self.RDE_VOS('encode_value', Fs[:,0], kf16[:,0], pgt2[:,0], pgt1[:,0])

                # value consistency perturbation branch
                v1_2 = self.RDE_VOS('encode_value', Fs[:,0], kf16[:,0], Ms[:,0], sec_Ms[:,0])
                v2_2 = self.RDE_VOS('encode_value', Fs[:,0], kf16[:,0], sec_Ms[:,0], Ms[:,0])

                ref_v = torch.stack([ref_v1, ref_v2], 1)

                # Segment frame 1
                f1_logits, f1_mask, _, _ = self.RDE_VOS('segment', 
                        k16[:,:,1], kf16_thin[:,1], kf8[:,1], kf4[:,1], 
                        k16[:,:,0:1], ref_v, selector)
                
                prev_v1 = self.RDE_VOS('encode_value', Fs[:,1], kf16[:,1], f1_mask[:,0:1], f1_mask[:,1:2])
                prev_v2 = self.RDE_VOS('encode_value', Fs[:,1], kf16[:,1], f1_mask[:,1:2], f1_mask[:,0:1])
                prev_v = torch.stack([prev_v1, prev_v2], 1) # [4, 2, 512, 1, 24, 24]
                values = torch.cat([ref_v, prev_v], 3) # [4, 2, 512, 2, 24, 24] N, O, C, T, H, W
                del prev_v
                del ref_v

                # compress memory bank for frame 2
                k2_thin, v2_thin = self.RDE_VOS('compress', k16[:,:,0:2], values)
                f2_thin_logits, f2_thin_mask, f2_thin_mv2qv_o1, f2_thin_mv2qv_o2= self.RDE_VOS('segment', 
                        k16[:,:,2], kf16_thin[:,2], kf8[:,2], kf4[:,2], 
                        k2_thin, v2_thin, selector)


                # Segment frame 2
                _, f2_mask, f2_mv2qv_o1, f2_mv2qv_o2 = self.RDE_VOS('segment', 
                        k16[:,:,2], kf16_thin[:,2], kf8[:,2], kf4[:,2], 
                        k16[:,:,0:2], values, selector)
                
                prev_v1 = self.RDE_VOS('encode_value', Fs[:,2], kf16[:,2], f2_mask[:,0:1], f2_mask[:,1:2])
                prev_v2 = self.RDE_VOS('encode_value', Fs[:,2], kf16[:,2], f2_mask[:,1:2], f2_mask[:,0:1])
                prev_v = torch.stack([prev_v1, prev_v2], 1) # [4, 2, 512, 1, 24, 24]
                values = torch.cat([values, prev_v], 3) # [4, 2, 512, 2, 24, 24] N, O, C, T, H, W
                del prev_v

                # Segment frame 3
                f3_logits, f3_mask, _, _ = self.RDE_VOS('segment', 
                        k16[:,:,3], kf16_thin[:,3], kf8[:,3], kf4[:,3], 
                        k16[:,:,0:3], values, selector)
                
                prev_v1 = self.RDE_VOS('encode_value', Fs[:,3], kf16[:,3], f3_mask[:,0:1], f3_mask[:,1:2])
                prev_v2 = self.RDE_VOS('encode_value', Fs[:,3], kf16[:,3], f3_mask[:,1:2], f3_mask[:,0:1])
                prev_v = torch.stack([prev_v1, prev_v2], 1) # [4, 2, 512, 1, 24, 24]
                values = torch.cat([values, prev_v], 3) # [4, 2, 512, 2, 24, 24] N, O, C, T, H, W
                del prev_v

                ##############################################
                # compress memory bank for frame 4
                # round-based propogation - frame 2
                k2_thin3 = torch.cat((k2_thin, k16[:,:,2:3]), dim=2)
                v2_thin3 = torch.cat((v2_thin, values[:,:,:,2:3]), dim=3)
                k3_thin, v3_thin = self.RDE_VOS('compress', k2_thin3, v2_thin3)

                # round-based propogation - frame 3
                k3_thin4 = torch.cat((k3_thin, k16[:,:,3:4]), dim=2)
                v3_thin4 = torch.cat((v3_thin, values[:,:,:,3:4]), dim=3)

                del k2_thin3, v2_thin3
                k4_thin, v4_thin = self.RDE_VOS('compress', k3_thin4, v3_thin4)

                del k3_thin4, v3_thin4
                ##############################################




                f4_thin_logits, f4_thin_mask, f4_thin_mv2qv_o1, f4_thin_mv2qv_o2 = self.RDE_VOS('segment', 
                        k16[:,:,4], kf16_thin[:,4], kf8[:,4], kf4[:,4], 
                        k4_thin, v4_thin, selector)
                
                del k4_thin, v4_thin
                # # Segment frame 4
                _, _, f4_mv2qv_o1, f4_mv2qv_o2 = self.RDE_VOS('segment', 
                        k16[:,:,4], kf16_thin[:,4], kf8[:,4], kf4[:,4], 
                        k16[:,:,0:4], values, selector)

                out['mask_1'] = f1_mask[:,0:1]
                out['mask_2'] = f2_thin_mask[:,0:1]
                out['mask_3'] = f3_mask[:,0:1]
                out['mask_4'] = f4_thin_mask[:,0:1]
                out['sec_mask_1'] = f1_mask[:,1:2]
                out['sec_mask_2'] = f2_thin_mask[:,1:2]
                out['sec_mask_3'] = f3_mask[:,1:2]
                out['sec_mask_4'] = f4_thin_mask[:,1:2]

                out['logits_1'] = f1_logits
                out['logits_2'] = f2_thin_logits
                out['logits_3'] = f3_logits
                out['logits_4'] = f4_thin_logits

                # del f1_mask, f2_thin_mask, f3_mask, f4_thin_mask
                # del f1_logits, f2_thin_logits, f3_logits, f4_thin_logits


            if self._do_log or self._is_train:
                losses = self.loss_computer.compute({**data, **out}, it)
                if not self.single_object:
                    # memory reader feature consistency
                    klloss = (self.klloss(ref_v1, v1_2) + self.klloss(ref_v2, v2_2)) * self.para['klloss_weight']
                    losses.update({'klloss_loss':klloss})
                    losses['total_loss'] = losses['total_loss'] +  klloss

                    # distillate the decoder input
                    decoder_f2_distillation_k_loss = (self.klloss(f2_thin_mv2qv_o1, f2_mv2qv_o1) + self.klloss(f2_thin_mv2qv_o2, f2_mv2qv_o2)) * self.para['decoder_f2_weight']
                    decoder_f4_distillation_k_loss = (self.klloss(f4_thin_mv2qv_o1, f4_mv2qv_o1) + self.klloss(f4_thin_mv2qv_o2, f4_mv2qv_o2)) * self.para['decoder_f4_weight']
                    losses.update({'df2_distillation':decoder_f2_distillation_k_loss})
                    losses.update({'df4_distillation':decoder_f4_distillation_k_loss})
                    losses['total_loss'] = losses['total_loss'] +  decoder_f2_distillation_k_loss + decoder_f4_distillation_k_loss
                else:
                    # memory reader feature consistency
                    klloss = self.klloss(ref_v, v_2) * self.para['klloss_weight']
                    losses.update({'klloss_loss':klloss})
                    losses['total_loss'] = losses['total_loss'] +  klloss

                # Logging
                if self._do_log:
                    self.integrator.add_dict(losses)
                    if self._is_train:
                        if it % self.save_im_interval == 0 and it != 0:
                            if self.logger is not None:
                                images = {**data, **out}
                                size = (384, 384)
                                self.logger.log_cv2('train/pairs', pool_pairs(images, size, self.single_object), it)

            if self._is_train:
                if (it) % self.report_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                        self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.report_interval, it)
                    self.last_time = time.time()
                    self.train_integrator.finalize('train', it)
                    self.train_integrator.reset_except_hooks()

                if it % self.save_model_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save(it)
                    # if self.para['stage']==2 or self.para['stage']==3:
                    #     try:
                    #         self.infer_model(it)
                    #     except:
                    #         pass
                # self.uploadLogger()

            # Backward pass
            # This should be done outside autocast
            # but I trained it like this and it worked fine
            # so I am keeping it this way for reference
            self.optimizer.zero_grad(set_to_none=True)
            if self.para['amp']:
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total_loss'].backward() 
                self.optimizer.step()
            self.scheduler.step()

    def save(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = self.save_path + ('_%s.pth' % it)
        torch.save(self.RDE_VOS.module.state_dict(), model_path)
        print('Model saved to %s.' % model_path)

        self.save_checkpoint(it)

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = self.save_path + '_checkpoint.pth'
        checkpoint = { 
            'it': it,
            'network': self.RDE_VOS.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)

        print('Checkpoint saved to %s.' % checkpoint_path)

    def load_model(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        self.RDE_VOS.module.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Model loaded.')

        return it

    def load_network(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = 'cuda:%d' % self.local_rank
        src_dict = torch.load(path, map_location={'cuda:0': map_location})

        # Maps SO weight (without other_mask) to MO weight (with other_mask)
        for k in list(src_dict.keys()):
            if k == 'value_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    pads = torch.zeros((64,1,7,7), device=src_dict[k].device)
                    nn.init.orthogonal_(pads)
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)

        self.RDE_VOS.module.load_state_dict(src_dict, strict=False)
        print('Network weight loaded:', path)

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        # Shall be in eval() mode to freeze BN parameters
        self.RDE_VOS.eval()
        if self.para['stage']==2 or self.para['stage']==3:
            self.RDE_VOS.module.mem_compress.train(True)
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.RDE_VOS.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.RDE_VOS.eval()
        return self

