

import numpy as np
import torch
import torch.nn as nn
import json
from data import SRN
from utils import get_rays, sample_from_rays, volume_rendering, image_float_to_uint8
from skimage.metrics import structural_similarity as compute_ssim
from model import CodeNeRF
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import imageio
import time


class Optimizer():
    def __init__(self, saved_dir, gpu, instance_ids=[], splits='test',
                 jsonfile = 'srncar.json', batch_size=2048, num_opts = 200):
        """
        :param saved_dir: the directory of pre-trained model
        :param gpu: which GPU we would use
        :param instance_id: the number of images for test-time optimization(ex : 000082.png)
        :param splits: test or val
        :param jsonfile: where the hyper-parameters are saved
        :param num_opts : number of test-time optimization steps
        """
        super().__init__()
        # Read Hyperparameters
        hpampath = os.path.join('jsonfiles', jsonfile)
        with open(hpampath, 'r') as f:
            self.hpams = json.load(f)
        self.device = torch.device('cuda:' + str(gpu))
        self.make_model()
        self.load_model_codes(saved_dir)
        self.make_dataloader(splits, len(instance_ids))
        print('we are going to save at ', self.save_dir)
        #self.saved_dir = saved_dir
        self.B = batch_size
        self.num_opts = num_opts
        self.splits = splits
        self.nviews = str(len(instance_ids))
        self.psnr_eval = {}
        self.psnr_opt = {}
        self.ssim_eval = {}


    def optimize_objs(self, instance_ids, lr=1e-2, lr_half_interval=50, save_img = True):
        logpath = os.path.join(self.save_dir, 'opt_hpams.json')
        hpam = {'instance_ids' : instance_ids, 'lr': lr, 'lr_half_interval': lr_half_interval,
                '
                
                ': self.splits}
        with open(logpath, 'w') as f:
            json.dump(hpam, f, indent=2)

        self.lr, self.lr_half_interval, iters = lr, lr_half_interval, 0
        instance_ids = torch.tensor(instance_ids)
        self.optimized_shapecodes = torch.zeros(len(self.dataloader), self.mean_shape.shape[1])
        self.optimized_texturecodes = torch.zeros(len(self.dataloader), self.mean_texture.shape[1])
        # Per object
        for num_obj, d in enumerate(self.dataloader):
            focal, H, W, imgs, poses, obj_idx = d
            tgt_imgs, tgt_poses = imgs[0, instance_ids], poses[0, instance_ids]
            self.nopts, self.lr_half_interval = 0, lr_half_interval
            shapecode = self.mean_shape.to(self.device).clone().detach().requires_grad_()
            texturecode = self.mean_texture.to(self.device).clone().detach().requires_grad_()

            # First Optimize
            self.set_optimizers(shapecode, texturecode)
            while self.nopts < self.num_opts:
                self.opts.zero_grad()
                t1 = time.time()
                generated_imgs, gt_imgs = [], []
                for num, instance_id in enumerate(instance_ids):
                    tgt_img, tgt_pose = tgt_imgs[num].reshape(-1,3), tgt_poses[num]
                    rays_o, viewdir = get_rays(H.item(), W.item(), focal, tgt_pose)
                    xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir, self.hpams['near'], self.hpams['far'],
                                                            self.hpams['N_samples'])
                    loss_per_img, generated_img = [], []
                    for i in range(0, xyz.shape[0], self.B):
                        sigmas, rgbs = self.model(xyz[i:i+self.B].to(self.device),
                                                  viewdir[i:i+self.B].to(self.device),
                                                  shapecode, texturecode)
                        rgb_rays, _ = volume_rendering(sigmas, rgbs, z_vals.to(self.device))
                        #print(rgb_rays.shape, tgt_img.shape)
                        loss_l2 = torch.mean((rgb_rays - tgt_img[i:i+self.B].type_as(rgb_rays))**2)
                        if i == 0:
                            reg_loss = torch.norm(shapecode, dim=-1) + torch.norm(texturecode, dim=-1)
                            loss_reg = self.hpams['loss_reg_coef'] * torch.mean(reg_loss)
                            loss = loss_l2 + loss_reg
                        else:
                            loss = loss_l2
                        loss.backward()
                        loss_per_img.append(loss_l2.item())
                        generated_img.append(rgb_rays)
                    generated_imgs.append(torch.cat(generated_img).reshape(H,W,3))
                    gt_imgs.append(tgt_img.reshape(H,W,3))
                self.opts.step()
                self.log_opt_psnr_time(np.mean(loss_per_img), time.time() - t1, self.nopts + self.num_opts * num_obj,
                                       num_obj)
                self.log_regloss(reg_loss.item(), self.nopts, num_obj)
                if self.save_img:
                    self.save_img(generated_imgs, gt_imgs, self.ids[num_obj], self.nopts)
                self.nopts += 1
                if self.nopts % lr_half_interval == 0:
                    self.set_optimizers(shapecode, texturecode)

            # Then, Evaluate
            with torch.no_grad():
                #print(tgt_poses.shape)
                for num in range(250):
                    if num not in instance_ids:
                        tgt_img, tgt_pose = imgs[0,num].reshape(-1,3), poses[0, num]
                        rays_o, viewdir = get_rays(H.item(), W.item(), focal, poses[0, num])
                        xyz, viewdir, z_vals = sample_from_rays(rays_o, viewdir, self.hpams['near'], self.hpams['far'],
                                                               self.hpams['N_samples'])
                        loss_per_img, generated_img = [], []
                        for i in range(0, xyz.shape[0], self.B):
                            sigmas, rgbs = self.model(xyz[i:i+self.B].to(self.device),
                                                      viewdir[i:i + self.B].to(self.device),
                                                      shapecode, texturecode)
                            rgb_rays, _ = volume_rendering(sigmas, rgbs, z_vals.to(self.device))
                            loss_l2 = torch.mean((rgb_rays - tgt_img[i:i+self.B].type_as(rgb_rays)) ** 2)
                            loss_per_img.append(loss_l2.item())
                            generated_img.append(rgb_rays)
                        self.log_eval_psnr(np.mean(loss_per_img), num, num_obj)
                        self.log_compute_ssim(torch.cat(generated_img).reshape(H, W, 3), tgt_img.reshape(H, W, 3),
                                              num, num_obj)
                        if save_img:
                            self.save_img([torch.cat(generated_img).reshape(H,W,3)], [tgt_img.reshape(H,W,3)], self.ids[num_obj], num,
                                          opt=False)

            # Save the optimized codes
            self.optimized_shapecodes[num_obj] = shapecode.detach().cpu()
            self.optimized_texturecodes[num_obj] = texturecode.detach().cpu()
            self.save_opts(num_obj)

    def save_opts(self, num_obj):
        saved_dict = {
            'ids': self.ids,
            'num_obj' : num_obj,
            'optimized_shapecodes' : self.optimized_shapecodes,
            'optimized_texturecodes': self.optimized_texturecodes,
            'psnr_eval': self.psnr_eval,
            'ssim_eval': self.ssim_eval
        }
        torch.save(saved_dict, os.path.join(self.save_dir, 'codes.pth'))
        print('We finished the optimization of ' + str(num_obj))

    def save_img(self, generated_imgs, gt_imgs, obj_id, instance_num, opt=True):
        H, W = gt_imgs[0].shape[:2]
        nviews = int(self.nviews)
        if not opt:
            nviews = 1
        generated_imgs = torch.cat(generated_imgs).reshape(nviews, H, W, 3)
        gt_imgs = torch.cat(gt_imgs).reshape(nviews, H, W, 3)
        ret = torch.zeros(nviews *H, 2 * W, 3)
        ret[:,:W,:] = generated_imgs.reshape(-1, W, 3)
        ret[:,W:,:] = gt_imgs.reshape(-1, W, 3)
        ret = image_float_to_uint8(ret.detach().cpu().numpy())
        save_img_dir = os.path.join(self.save_dir, obj_id)
        if not os.path.isdir(save_img_dir):
            os.makedirs(save_img_dir)
        if opt:
            imageio.imwrite(os.path.join(save_img_dir, 'opt' + self.nviews + '_' + str(instance_num) + '.png'), ret)
        else:
            imageio.imwrite(os.path.join(save_img_dir, str(instance_num) + '_' + self.nviews + '.png'), ret)

    def log_compute_ssim(self, generated_img, gt_img, niters, obj_idx):
        generated_img_np = generated_img.detach().cpu().numpy()
        gt_img_np = gt_img.detach().cpu().numpy()
        ssim = compute_ssim(generated_img_np, gt_img_np, multichannel=True)
        if niters == 0:
            self.ssim_eval[obj_idx] = [ssim]
        else:
            self.ssim_eval[obj_idx].append(ssim)

    def log_eval_psnr(self, loss_per_img, niters, obj_idx):
        psnr = -10 * np.log(loss_per_img) / np.log(10)
        if niters == 0:
            self.psnr_eval[obj_idx] = [psnr]
        else:
            self.psnr_eval[obj_idx].append(psnr)

    def log_opt_psnr_time(self, loss_per_img, time_spent, niters, obj_idx):
        psnr = -10*np.log(loss_per_img) / np.log(10)
        self.writer.add_scalar('psnr_opt/' + self.nviews + '/' + self.splits, psnr, niters, obj_idx)
        self.writer.add_scalar('time_opt/' + self.nviews + '/' + self.splits, time_spent, niters, obj_idx)

    def log_regloss(self, loss_reg, niters, obj_idx):
        self.writer.add_scalar('reg/'  + self.nviews + '/' + self.splits, loss_reg, niters, obj_idx)

    def set_optimizers(self, shapecode, texturecode):
        lr = self.get_learning_rate()
        #print(lr)
        self.opts = torch.optim.AdamW([
            {'params': shapecode, 'lr': lr},
            {'params': texturecode, 'lr':lr}
        ])

    def get_learning_rate(self):
        opt_values = self.nopts // self.lr_half_interval
        lr = self.lr * 2**(-opt_values)
        return lr

    def make_model(self):
        self.model = CodeNeRF(**self.hpams['net_hyperparams']).to(self.device)

    def load_model_codes(self, saved_dir):
        saved_path = os.path.join('exps', saved_dir, 'models.pth')
        saved_data = torch.load(saved_path, map_location = torch.device('cpu'))
        self.make_save_img_dir(os.path.join('exps', saved_dir, 'test'))
        self.make_writer(saved_dir)
        self.model.load_state_dict(saved_data['model_params'])
        self.model = self.model.to(self.device)
        self.mean_shape = torch.mean(saved_data['shape_code_params']['weight'], dim=0).reshape(1,-1)
        self.mean_texture = torch.mean(saved_data['texture_code_params']['weight'], dim=0).reshape(1,-1)

    def make_writer(self, saved_dir):
        self.writer = SummaryWriter(os.path.join('exps', saved_dir, 'test', 'runs'))

    def make_save_img_dir(self, save_dir):
        save_dir_tmp = save_dir
        num = 2
        while os.path.isdir(save_dir_tmp):
            save_dir_tmp = save_dir + '_' + str(num)
            num += 1

        os.makedirs(save_dir_tmp)
        self.save_dir = save_dir_tmp
        #print(self.save_dir)

    def make_dataloader(self, splits, num_instances_per_obj, crop_img=False):
        cat = self.hpams['data']['cat']
        data_dir = self.hpams['data']['data_dir']
        obj = cat.split('_')[1]
        splits = obj + '_' + splits
        srn = SRN(cat=cat, splits=splits, data_dir = data_dir,
                  num_instances_per_obj = num_instances_per_obj, crop_img = crop_img)
        self.ids = srn.ids
        self.dataloader = DataLoader(srn, batch_size=1, num_workers =4, shuffle = False)

