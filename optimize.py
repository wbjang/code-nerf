
import sys, os

ROOT_DIR = os.path.abspath(os.path.join('', 'src'))
sys.path.insert(0, os.path.join(ROOT_DIR))

import argparse
from utils import str2bool
from optimizer import Optimizer



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="CodeNeRF")
    arg_parser.add_argument("--gpu",dest="gpu",required=True)
    arg_parser.add_argument("--saved_dir", dest="saved_dir", required=True)
    arg_parser.add_argument("--tgt_instances", dest = "tgt_instances", nargs='+', required = True)
    arg_parser.add_argument("--splits", dest="splits", default='test')
    arg_parser.add_argument("--num_opts", dest="num_opts", default = 200)
    arg_parser.add_argument("--lr", dest="lr", default=1e-2)
    arg_parser.add_argument("--lr_half_interval", dest="lr_half_interval", default=50)
    arg_parser.add_argument("--save_img", dest="save_img", default=True)
    arg_parser.add_argument("--jsonfile", dest="jsonfile", default="srncar.json")
    arg_parser.add_argument("--batchsize", dest="batchsize", default=2048)

    args = arg_parser.parse_args()
    saved_dir = args.saved_dir
    gpu = int(args.gpu)
    lr = float(args.lr)
    lr_half_interval = int(args.lr_half_interval)
    save_img = str2bool(args.save_img)
    batchsize = int(args.batchsize)
    tgt_instances = list(args.tgt_instances)
    num_opts = int(args.num_opts)
    for num, i in enumerate(tgt_instances):
        tgt_instances[num] = int(i)
    optimizer = Optimizer(saved_dir, gpu, tgt_instances, args.splits, args.jsonfile, batchsize, num_opts)
    optimizer.optimize_objs(tgt_instances, lr, lr_half_interval, save_img)

