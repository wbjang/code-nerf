


## CodeNeRF: Disentangled Neural Radiance Fields for Object Categories

Date : 27th Feb, 2022

This contains the implementation of the paper [CodeNeRF](https://arxiv.org/abs/2109.01750). 
Please refer to the [project webpage](https://sites.google.com/view/wbjang/home/codenerf) for demos.


### Install the environment


```
conda env create -f environment.yml
conda activate code_nerf
```

### Catalog

- [x] Training
- [x] Optimizing with GT pose
- [ ] Editing Shapes/Textures
- [ ] Pose Optimizing


### Download the data (ShapeNet-SRN)

For ShapeNet-SRN dataset, you can download it from https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR

### Training

```
python train.py --gpu <gpu_id> --save_dir <save_dir> --jsonfiles <jsonfile.json> --iters_crop 1000000 --iters_all 1200000
```

JSON files contain hyper-parameters as well as data directory. 'iters_crop' and 'iters_all' are number of iterations for both cropped and whole images.

### Optimizing

```
python optimize.py --gpu <gpu_id> --saved_dir <trained_dir>
```

The result will be stored in <trained_dir/test(_num)>, and each folder contains the progress of optimization, and the evaluation of test set. 
The final optimized results and the quantitative evaluations are stored in 'trained_dir/test(_num)/codes.pth'




### BibTex

```
@inproceedings{jang2021codenerf,
  title={Codenerf: Disentangled neural radiance fields for object categories},
  author={Jang, Wonbong and Agapito, Lourdes},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12949--12958},
  year={2021}
}
```

### References

Some parts of code are borrowed from below amazing repositories.

* The GitHub repository of Pixel NeRF : https://github.com/sxyu/pixel-nerf
* nerf_pl : https://github.com/kwea123/nerf_pl
* NeRF-pytorch : https://github.com/yenchenlin/nerf-pytorch


### Supplementary Video

https://user-images.githubusercontent.com/32883157/130004248-0ff74d4e-993e-43f2-91ee-bd25776e65bc.mp4


### License

MIT

