
import argparse
import os
from PIL import Image
import torchvision.transforms as transforms
import time
from configuration import *
from torch.utils.data import DataLoader
from datasets.Dataset_loader import Dataset_test
from tqdm import tqdm
from utils import *
from model.model import My_model
import losses
import logging
from tensorboardX import SummaryWriter


def test(args):
    os.environ['CUDA_VISIBLE_DEVICES']=args.CUDA_VISIBLE_DEVICES
    os.environ['CUDA_LAUNCH_BLOCKING']=args.CUDA_VISIBLE_DEVICES
    import torch;
    args.state ='test';
    test_dataset=Dataset_test(args,phase="test");
    test_loader=DataLoader(
        test_dataset,
        batch_size=args.batch_size_test,
        shuffle=True,
        drop_last=False,
        num_workers=8,
        pin_memory=True
    )
    event_path=make_dirs(args);
    logger = configure_logging(event_path)
    writer=SummaryWriter(os.path.join(event_path,'event_path'))
    # TODO 填入网络模型名字 填写是否pretrain
    model=My_model().cuda();
    ckpt = torch.load('./experiment/train/.../best_encoder.pth.tar');
    model.load_state_dict(ckpt["state_dict"]);
    
    # TODO 填写选择的loss
    loss_computer=losses.my_loss().eval();
    # TODO 填写说明
    logging.info("说明")
    loss_test=[];
    model.eval();
    for i, input_dict in tqdm(enumerate(test_loader)):
        input_dict = dict_to_cuda(input_dict);
        # TODO 将input_dict中的内容输入到网络中
        # 并取出gt做监督

        # eg:
        # image_a=input_dict["inputA"].cuda().float();
        # image_b=input_dict["inputB"].cuda().float();
        # detail_pts_gt=input_dict["pts"].cuda().float();
        # coarse_pts_gt=input_dict["coarse_pts"].cuda().float();
        # input_tensor=torch.cat([image_a,image_b],dim=1);
        # output_para = model(input_tensor);
        # TODO 计算loss
        # loss=loss_computer(input,gt).requires_grad_();

        logging.info("finish infernence {}".format(input_dict["basename"]))
        
        loss_test.append(loss.item());
        if i>0 and (i % args.writesummary == 0):
            logging.info(
                'test iter:{},loss:{:.5f}'.format(
                    i,
                    torch.tensor(loss_test[-1]),
                ))
        if args.test_save:
            if i<10:
                # TODO 完成重建结果的可视化保存
                # 如mesh、图片等
                pass
    logging.info("average loss {}".format(torch.tensor(loss_test).mean()))
    return
def main(args):
    test(args);
    return
if __name__ == '__main__':
    args.CUDA_VISIBLE_DEVICES="2"
    main(args)
    
