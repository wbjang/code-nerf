
import imageio
import numpy as np
import torch
# import json
# from torchvision import transforms
import os


def load_poses(pose_dir, idxs=[]):
    txtfiles = np.sort([os.path.join(pose_dir, f.name) for f in os.scandir(pose_dir)])
    posefiles = np.array(txtfiles)[idxs]
    srn_coords_trans = np.diag(np.array([1, -1, -1, 1])) # SRN dataset
    poses = []
    for posefile in posefiles:
        pose = np.loadtxt(posefile).reshape(4,4)
        poses.append(pose@srn_coords_trans)
    return torch.from_numpy(np.array(poses)).float()

def load_imgs(img_dir, idxs = []):
    allimgfiles = np.sort([os.path.join(img_dir, f.name) for f in os.scandir(img_dir)])
    imgfiles = np.array(allimgfiles)[idxs]
    imgs = []
    for imgfile in imgfiles:
        img = imageio.imread(imgfile, pilmode='RGB')
        img = img.astype(np.float32)
        img /= 255.
        imgs.append(img)
    return torch.from_numpy(np.array(imgs))

def load_intrinsic(intrinsic_path):
    with open(intrinsic_path, 'r') as f:
        lines = f.readlines()
        focal = float(lines[0].split()[0])
        H, W = lines[-1].split()
        H, W = int(H), int(W)
    return focal, H, W

class SRN():
    def __init__(self, cat='srn_cars', splits='cars_train',
                 data_dir = '../data/ShapeNet_SRN/',
                num_instances_per_obj = 1, crop_img = True):
        """
        cat: srn_cars / srn_chairs
        split: cars_train(/test/val) or chairs_train(/test/val)
        First, we choose the id
        Then, we sample images (the number of instances matter)
        """
        self.data_dir = os.path.join(data_dir, cat, splits)
        self.ids = np.sort([f.name for f in os.scandir(self.data_dir)])
        self.lenids = len(self.ids)
        self.num_instances_per_obj = num_instances_per_obj
        self.train = True if splits.split('_')[1] == 'train' else False
        self.crop_img = crop_img

    def __len__(self):
        return self.lenids
    
    def __getitem__(self, idx):
        obj_id = self.ids[idx]
        if self.train:
            focal, H, W, imgs, poses, instances = self.return_train_data(obj_id)
            return focal, H, W, imgs, poses, instances, idx
        else:
            focal, H, W, imgs, poses = self.return_test_val_data(obj_id)
            return focal, H, W, imgs, poses, idx
    
    def return_train_data(self, obj_id):
        pose_dir = os.path.join(self.data_dir, obj_id, 'pose')
        img_dir = os.path.join(self.data_dir, obj_id, 'rgb')
        intrinsic_path = os.path.join(self.data_dir, obj_id, 'intrinsics.txt')
        instances = np.random.choice(50, self.num_instances_per_obj)
        poses = load_poses(pose_dir, instances)
        imgs = load_imgs(img_dir, instances)
        focal, H, W = load_intrinsic(intrinsic_path)
        if self.crop_img:
            imgs = imgs[:,32:-32,32:-32,:]
            H, W = H // 2, W//2
        return focal, H, W, imgs.reshape(self.num_instances_per_obj, -1,3), poses, instances
    
    def return_test_val_data(self, obj_id):
        pose_dir = os.path.join(self.data_dir, obj_id, 'pose')
        img_dir = os.path.join(self.data_dir, obj_id, 'rgb')
        intrinsic_path = os.path.join(self.data_dir, obj_id, 'intrinsics.txt')
        instances = np.arange(250)
        poses = load_poses(pose_dir, instances)
        imgs = load_imgs(img_dir, instances)
        focal, H, W = load_intrinsic(intrinsic_path)
        return focal, H, W, imgs, poses