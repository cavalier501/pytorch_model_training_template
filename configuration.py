import argparse
import numpy as np
import math

parse = argparse.ArgumentParser('configration of program')
parse.add_argument('--CUDA_VISIBLE_DEVICES',type=str,default="0")

data_parse = parse.add_argument_group("dataset")
data_parse.add_argument('--imagesize',type=int,default=256)

logger_parse = parse.add_argument_group("logger")
logger_parse.add_argument('--is_logger',type=bool,default=True)
logger_parse.add_argument('--event_path',type=str,default='./experiment')
logger_parse.add_argument('--iswriter',type=bool,default=False)

train_parse = parse.add_argument_group("train")
train_parse.add_argument('--batch_size',type=int,default=8)
train_parse.add_argument('--lr',type=float,default=2e-4)
train_parse.add_argument('--num_epochs',type=int,default=400)
train_parse.add_argument('--writesummary',type=int,default=20)
train_parse.add_argument('--pre_train', type=bool, default=False)
train_parse.add_argument('--pre_train_ckpt', type=str, default='')
train_parse.add_argument('--already_trained_epoch', type=int, default=0)

test_parse = parse.add_argument_group("test")
test_parse.add_argument('--batch_size_test',type=int,default=1)
test_parse.add_argument('--test_save',type=bool,default=True)

args = parse.parse_args()
args.milestones=[50,100,150,200]

