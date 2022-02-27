
import imageio
import numpy as np
import torch
# import json
# from torchvision import transforms
import os


def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :].type_as(c2w) * c2w[..., :3, :3], -1)
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = c2w[..., :3, -1].expand(rays_d.shape)
    rays_o, viewdirs = rays_o.reshape(-1, 3), viewdirs.reshape(-1, 3)
    return rays_o, viewdirs

def sample_from_rays(ro, vd, near, far, N_samples, z_fixed = False):
    # Given ray centre (camera location), we sample z_vals
    # we do not use ray_o here - just number of rays
    if z_fixed:
        z_vals = torch.linspace(near, far, N_samples).type_as(ro)
    else:
        dist = (far - near) / (2*N_samples)
        z_vals = torch.linspace(near+dist, far-dist, N_samples).type_as(ro)
        z_vals += torch.rand(N_samples) * (far - near) / (2*N_samples)
    xyz = ro.unsqueeze(-2) + vd.unsqueeze(-2) * z_vals.unsqueeze(-1)
    vd = vd.unsqueeze(-2).repeat(1,N_samples,1)
    return xyz, vd, z_vals

def volume_rendering(sigmas, rgbs, z_vals, white_bg = True):
    deltas = z_vals[1:] - z_vals[:-1]
    deltas = torch.cat([deltas, torch.ones_like(deltas[:1]) * 1e10])
    alphas = 1 - torch.exp(-sigmas.squeeze(-1) * deltas)
    trans = 1 - alphas + 1e-10
    transmittance = torch.cat([torch.ones_like(trans[..., :1]), trans], -1)
    accum_trans = torch.cumprod(transmittance, -1)[..., :-1]
    weights = alphas * accum_trans
    rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)
    depth_final = torch.sum(weights * z_vals, -1)
    if white_bg:
        weights_sum = weights.sum(1)
        rgb_final = rgb_final + 1 - weights_sum.unsqueeze(-1)
    return rgb_final, depth_final

def image_float_to_uint8(img):
    """
    Convert a float image (0.0-1.0) to uint8 (0-255)
    """
    #print(img.shape)
    vmin = np.min(img)
    vmax = np.max(img)
    if vmax - vmin < 1e-10:
        vmax += 1e-10
    img = (img - vmin) / (vmax - vmin)
    img *= 255.0
    return img.astype(np.uint8)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true'):
        return True
    elif v.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')