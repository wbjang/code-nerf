
import sys, os

ROOT_DIR = os.path.abspath(os.path.join('', 'src'))
sys.path.insert(0, os.path.join(ROOT_DIR))

import argparse

from trainer import Trainer



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="CodeNeRF")
    arg_parser.add_argument("--gpu",dest="gpu",required=True)
    arg_parser.add_argument("--save_dir", dest="save_dir", required=True)
    arg_parser.add_argument("--iters_crop", dest = "iters_crop", required = True)
    arg_parser.add_argument("--iters_all", dest="iters_all", required=True)
    arg_parser.add_argument("--batchsize", dest="batchsize", default = 2048)
    arg_parser.add_argument("--jsonfile", dest="jsonfile", default="srncar.json")
    arg_parser.add_argument("--num_instances_per_obj", dest="num_instances_per_obj", default=2)

    args = arg_parser.parse_args()
    save_dir = args.save_dir
    gpu = int(args.gpu)
    iters_crop = int(args.iters_crop)
    iters_all = int(args.iters_all)
    B = int(args.batchsize)
    num_instances_per_obj = int(args.num_instances_per_obj)

    trainer = Trainer(save_dir, gpu, jsonfile = args.jsonfile, batch_size = B)
    trainer.training(iters_crop, iters_all, num_instances_per_obj)

