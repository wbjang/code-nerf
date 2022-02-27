

import numpy as np
import torch
import torch.nn as nn
import json
from data import SRN
from utils import get_rays, sample_from_rays, volume_rendering, image_float_to_uint8
from model import CodeNeRF
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import math
import time


class Trainer():
    def __init__(self, save_dir, gpu, jsonfile = 'srncar.json', batch_size=2048,
                 check_iter = 10000):
        super().__init__()
        # Read Hyperparameters
        hpampath = os.path.join('jsonfiles', jsonfile)
        with open(hpampath, 'r') as f:
            self.hpams = json.load(f)
        self.device = torch.device('cuda:' + str(gpu))
        self.make_model()
        self.make_dataloader(num_instances_per_obj = 1, crop_img = False)
        self.make_codes()
        self.B = batch_size
        self.make_savedir(save_dir)
        self.niter, self.nepoch = 0, 0
        self.check_iter = check_iter


    def training(self, iters_crop, iters_all, num_instances_per_obj=1):
        if iters_crop > iters_all:
            raise error
        while self.niter < iters_all:
            if self.niter < iters_crop:
                self.training_single_epoch(num_instances_per_obj = num_instances_per_obj,
                                           num_iters = iters_crop, crop_img = True)
            else:
                self.training_single_epoch(num_instances_per_obj=num_instances_per_obj,
                                           num_iters=iters_all, crop_img = False)
            self.save_models()
            self.nepoch += 1

    def training_single_epoch(self, num_instances_per_obj, num_iters, crop_img = True):
        # single epoch here means that it iterates over whole objects
        # only 1 or a few images are chosen for each epoch
        self.make_dataloader(num_instances_per_obj, crop_img = crop_img)
        self.set_optimizers()
        # per object
        for d in self.dataloader:
            if self.niter < num_iters:
                focal, H, W, imgs, poses, instances, obj_idx = d
                obj_idx = obj_idx.to(self.device)
                # per image
                self.opts.zero_grad()
                for k in range(num_instances_per_obj):
                    # print(k, num_instances_per_obj, poses[0, k].shape, imgs.shape, 'k')
                    t1 = time.time()
                    self.opts.zero_grad()
                    rays_o, viewdir = get_rays(H.item(), W.item(), focal, poses[0,k])
                    xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir, self.hpams['near'], self.hpams['far'],
                                            self.hpams['N_samples'])
                    loss_per_img, generated_img = [], []
                    for i in range(0, xyz.shape[0], self.B):
                        shape_code, texture_code = self.shape_codes(obj_idx), self.texture_codes(obj_idx)
                        sigmas, rgbs = self.model(xyz[i:i+self.B].to(self.device),
                                                  viewdir[i:i+self.B].to(self.device),
                                                  shape_code, texture_code)
                        rgb_rays, _ = volume_rendering(sigmas, rgbs, z_vals.to(self.device))
                        loss_l2 = torch.mean((rgb_rays - imgs[0, k, i:i+self.B].type_as(rgb_rays))**2)
                        if i == 0:
                            reg_loss = torch.norm(shape_code, dim=-1) + torch.norm(texture_code, dim=-1)
                            loss_reg = self.hpams['loss_reg_coef'] * torch.mean(reg_loss)
                            loss = loss_l2 + loss_reg
                        else:
                            loss = loss_l2
                        loss.backward()
                        loss_per_img.append(loss_l2.item())
                        generated_img.append(rgb_rays)
                self.opts.step()
                self.log_psnr_time(np.mean(loss_per_img), time.time() - t1, obj_idx)
                self.log_regloss(reg_loss, obj_idx)
                if self.niter % self.check_iter == 0:
                    generated_img = torch.cat(generated_img)
                    generated_img = generated_img.reshape(H,W,3)
                    gtimg = imgs[0,-1].reshape(H,W,3)
                    self.log_img(generated_img, gtimg, obj_idx)
                    print(-10*np.log(np.mean(loss_per_img))/np.log(10), self.niter)
                if self.niter % self.hpams['check_points'] == 0:
                    self.save_models(self.niter)
                self.niter += 1

    def log_psnr_time(self, loss_per_img, time_spent, obj_idx):
        psnr = -10*np.log(loss_per_img) / np.log(10)
        self.writer.add_scalar('psnr/train', psnr, self.niter, obj_idx)
        self.writer.add_scalar('time/train', time_spent, self.niter, obj_idx)

    def log_regloss(self, loss_reg, obj_idx):
        self.writer.add_scalar('reg/train', loss_reg, self.niter, obj_idx)

    def log_img(self, generated_img, gtimg, obj_idx):
        H, W = generated_img.shape[:-1]
        ret = torch.zeros(H,2*W, 3)
        ret[:,:W,:] = generated_img
        ret[:,W:,:] = gtimg
        ret = image_float_to_uint8(ret.detach().cpu().numpy())
        self.writer.add_image('train_'+str(self.niter) + '_' + str(obj_idx.item()), torch.from_numpy(ret).permute(2,0,1))

    def set_optimizers(self):
        lr1, lr2 = self.get_learning_rate()
        self.opts = torch.optim.AdamW([
            {'params':self.model.parameters(), 'lr': lr1},
            {'params':self.shape_codes.parameters(), 'lr': lr2},
            {'params':self.texture_codes.parameters(), 'lr':lr2}
        ])

    def get_learning_rate(self):
        model_lr, latent_lr = self.hpams['lr_schedule'][0], self.hpams['lr_schedule'][1]
        num_model = self.niter // model_lr['interval']
        num_latent = self.niter // latent_lr['interval']
        lr1 = model_lr['lr'] * 2**(-num_model)
        lr2 = latent_lr['lr'] * 2**(-num_latent)
        return lr1, lr2

    def make_model(self):
        self.model = CodeNeRF(**self.hpams['net_hyperparams']).to(self.device)

    def make_codes(self):
        embdim = self.hpams['net_hyperparams']['latent_dim']
        d = len(self.dataloader)
        self.shape_codes = nn.Embedding(d, embdim)
        self.texture_codes = nn.Embedding(d, embdim)
        self.shape_codes.weight = nn.Parameter(torch.randn(d, embdim) / math.sqrt(embdim/2))
        self.texture_codes.weight = nn.Parameter(torch.randn(d, embdim) / math.sqrt(embdim/2))
        self.shape_codes = self.shape_codes.to(self.device)
        self.texture_codes = self.texture_codes.to(self.device)
        
    def make_dataloader(self, num_instances_per_obj, crop_img):
        # cat : whether it is 'srn_cars' or 'srn_chairs'
        # split: whether it is 'car_train', 'car_test' or 'car_val'
        # data_dir : the root directory of ShapeNet_SRN
        # num_instances_per_obj : how many images we chosose from objects
        cat = self.hpams['data']['cat']
        data_dir = self.hpams['data']['data_dir']
        splits = self.hpams['data']['splits']
        srn = SRN(cat=cat, splits=splits, data_dir = data_dir,
                  num_instances_per_obj = num_instances_per_obj, crop_img = crop_img)
        self.dataloader = DataLoader(srn, batch_size=1, num_workers =4)

    def make_savedir(self, save_dir):
        self.save_dir = os.path.join('exps', save_dir)
        if not os.path.isdir(self.save_dir):
            os.makedirs(os.path.join(self.save_dir, 'runs'))
        self.writer = SummaryWriter(os.path.join(self.save_dir, 'runs'))
        hpampath = os.path.join(self.save_dir, 'hpam.json')
        with open(hpampath, 'w') as f:
            json.dump(self.hpams, f, indent=2)


    def save_models(self, iter = None):
        save_dict = {'model_params': self.model.state_dict(),
                     'shape_code_params': self.shape_codes.state_dict(),
                     'texture_code_params': self.texture_codes.state_dict(),
                     'niter': self.niter,
                     'nepoch' : self.nepoch
                     }
        if iter != None:
            torch.save(save_dict, os.path.join(self.save_dir, str(iter) + '.pth'))
        torch.save(save_dict, os.path.join(self.save_dir, 'models.pth'))


