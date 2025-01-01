
import argparse
import os
from PIL import Image
import torchvision.transforms as transforms
import time
from configuration import *
from torch.utils.data import DataLoader
from datasets.Dataset_loader import Dataset_train
from datasets.Dataset_loader import Dataset_test
from tqdm import tqdm
from utils import *
from model.model import My_model
import losses
import logging
import matplotlib.pyplot as plt
import pylab as pl
from tensorboardX import SummaryWriter
"""
date:2023.11.19
note:
相比于train.py 有如下改动：
1 更改了保存checkpoint和loss曲线的频率：
只在出现最佳checkpoint时读取、保存最佳checkpoint(同时覆盖前一个最佳checkpoint)，不保存最新checkpoint（其实train.py也是这样的）
每100(100为超参数)个epoch保存一次loss曲线 但每个epoch仍要记录当前的loss
2 增加了“输出每个epoch的训练、测试集loss”的功能
train.py中只有输出每个batch的loss，当一个epoch中包含多个batch时，不方便loss的整理
暂定每个batch中batch_size相同(即dataloader中不存在样本数＜预定batch_size的情况)
"""



def main(args):
    os.environ['CUDA_VISIBLE_DEVICES']=args.CUDA_VISIBLE_DEVICES
    os.environ['CUDA_LAUNCH_BLOCKING']=args.CUDA_VISIBLE_DEVICES
    import torch;
    args.state ='train';
    train_dataset=Dataset_train(args,phase="train");
    train_loader=DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=8,
        pin_memory=True
    )
    test_dataset=Dataset_test(args,phase="test")
    test_loader=DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )
    event_path=make_dirs(args);
    logger = configure_logging(event_path)
    # setup_logging_and_parse_arguments(logger)    
    
    writer=SummaryWriter(os.path.join(event_path,'event_path'))

    # TODO 填入网络模型名字 填写是否pretrain
    model=My_model().cuda();
    if args.pre_train==True:
        args.already_trained_epoch=300;
        args.num_epochs=450
        ckpt = torch.load('')
        model.load_state_dict(ckpt["state_dict"],strict=True);

    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=0.00001, eps=1e-8);
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)
    already_trained_epoch=args.already_trained_epoch;
    writer=SummaryWriter(os.path.join(event_path,'event_path'));


    loss_train_per_epoch=[];
    loss_test_per_epoch=[];
    epoch_list=[];
    best_loss=0;
    old_ckpt_path=""
    # TODO 填写选择的loss
    loss_computer=losses.my_loss();

    # TODO 填写说明
    logging.info("说明")
    for epoch in tqdm(range(already_trained_epoch,args.num_epochs)):
        loss_train=[];
        loss_test =[];

        model.train();
        loss_computer.train();
        for i, input_dict in tqdm(enumerate(train_loader)):
            torch.autograd.set_detect_anomaly(True)
            iter = epoch * len(train_loader) + i
            input_dict = dict_to_cuda(input_dict);
            # TODO 将input_dict中的内容输入到网络中
            # 并取出gt做监督
            # eg:
            # image_a=input_dict["inputA"].cuda().float();
            # image_b=input_dict["inputB"].cuda().float();
            # detail_pts_gt=input_dict["pts"].cuda().float();
            # input_tensor=torch.cat([image_a,image_b],dim=1);
            # output_para = model(input_tensor);
            # id_para=output_para[:,0:50];exp_para=output_para[:,50:102];

            # TODO 计算loss
            # loss=loss_computer(input,gt)


            loss_train.append(loss.item());

            optimizer.zero_grad();
            loss.backward()
            optimizer.step()                



            if i>0 and (i % args.writesummary == 0 or i<5):
                logging.info(
                    (
                        f"train iter:{iter},lr rate:{scheduler.get_last_lr()[0]:.6f},"
                        f"loss:{loss.item():.5f}"
                    )
                )                  
        scheduler.step()

        for j,input_dict in tqdm(enumerate(test_loader)):
            with torch.no_grad():
                test_iter = epoch * len(test_loader) + j
                input_dict = dict_to_cuda(input_dict)
                # TODO 将input_dict中的内容输入到网络中
                # 并取出gt做监督      
                # eg：          
                # image_a=input_dict["inputA"].float();
                # image_b=input_dict["inputB"].float();
                # detail_pts_gt=input_dict["pts"];
                # input_tensor=torch.cat([image_a,image_b],dim=1);
                # output_para = model(input_tensor);
                # id_para=output_para[:,0:50];exp_para=output_para[:,50:102];
                
                # TODO 计算loss
                # loss=loss_computer(input,gt)
                loss_test.append(loss.item());
                if j>0 and j % args.writesummary == 0 :
                    logging.info(
                        (
                            f"test  iter:{test_iter},lr rate:{scheduler.get_last_lr()[0]:.6f},"
                            f"loss:{torch.tensor(loss_test).mean():.5f}"
                        )
                    )                         
        if args.iswriter:
            writer.add_scalars('loss', {"train_loss":torch.tensor(loss_train).mean(),
                                       "test_loss":torch.tensor(loss_test).mean()}, epoch)          

        if epoch ==already_trained_epoch:
            best_loss = torch.tensor(loss_test).mean()
        if best_loss >= torch.tensor(loss_test).mean():
            save_state = {'state_dict': model.state_dict()}
            ckpt_path = os.path.join(event_path, 'checkpoint_path')            
            best_loss = torch.tensor(loss_test).mean()
            logging.info("save best checkpoint!")
            ckpt_path = os.path.join(event_path, 'checkpoint_path')
            torch.save(save_state, os.path.join(ckpt_path, 'best_' + 'encoder.pth.tar'),_use_new_zipfile_serialization=False)
            torch.save(save_state, os.path.join(ckpt_path, 'latest_'+str(epoch) + '_encoder.pth.tar'),_use_new_zipfile_serialization=False)
            if os.path.exists(old_ckpt_path):
                os.remove(old_ckpt_path)
            old_ckpt_path=os.path.join(ckpt_path, 'latest_'+str(epoch) + '_encoder.pth.tar')

        logging.info("save latest {} checkpoint! ".format(str(epoch)))

        # draw
        loss_train_per_epoch.append(torch.tensor(loss_train).mean());
        loss_test_per_epoch.append(torch.tensor(loss_test).mean());
        epoch_list.append(epoch);
        for epoch_draw_iter in epoch_list:
            if loss_train_per_epoch[epoch_draw_iter-already_trained_epoch]>100*best_loss:
                loss_train_per_epoch[epoch_draw_iter-already_trained_epoch]=0
                loss_test_per_epoch[epoch_draw_iter-already_trained_epoch]=0
        if epoch%100==0:
            fig=plt.figure()
            pl.plot(epoch_list,loss_train_per_epoch,"r-",label="train_loss")
            pl.plot(epoch_list,loss_test_per_epoch,"g-",label="test_loss")
            pl.legend()
            pl.xlabel("epoch")
            pl.ylabel("loss")
            plt.savefig(os.path.join(ckpt_path,"loss.jpg"))  
        # TODO train_v2.py更新
        logging.info(f"train loss of epoch {epoch}: {torch.tensor(loss_train).mean():.5f}")      
        logging.info(f"test  loss of epoch {epoch}: {torch.tensor(loss_test).mean():.5f}")      
    pass



def print_model():
    import torch
    model = VggEncoder()
    model = torch.nn.DataParallel(model).cuda()
    # ckpt = torch.load('data/net.pth')
    # model.load_state_dict(ckpt)
    print(model)
    pass


if __name__=="__main__":
    args.CUDA_VISIBLE_DEVICES="0";
    main(args);
