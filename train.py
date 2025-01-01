
import argparse
import os
from PIL import Image
import torchvision.transforms as transforms
import time
from configuration import *
from torch.utils.data import DataLoader
from datasets.Dataset_loader import Dataset_train
from datasets.Dataset_loader import Dataset_val
from tqdm import tqdm
from utils import *
from model.model import My_model
import losses
import logging
from tensorboardX import SummaryWriter

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
    val_dataset=Dataset_val(args,phase="val")
    val_loader=DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
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
    loss_val_per_epoch=[];
    epoch_list=[];
    best_loss=0;
    old_ckpt_path=""
    # TODO 填写选择的loss
    loss_computer=losses.my_loss();

    # TODO 填写说明
    logging.info("说明")
    for epoch in range(already_trained_epoch,args.num_epochs):
        loss_train=[];
        loss_val =[];

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
                    'train iter:{},lr rate:{:.6f},loss:{:.5f}'.format(
                        iter,
                        scheduler.get_last_lr()[0],
                        torch.tensor(loss_train).mean(),
                    ))
        scheduler.step()

        for j,input_dict in tqdm(enumerate(val_loader)):
            with torch.no_grad():
                val_iter = epoch * len(val_loader) + j
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
                loss_val.append(loss.item());
                if j>0 and j % args.writesummary == 0 :
                    logging.info(
                        'val   iter:{},lr rate:{:.6f},loss:{:.5f}'.format(
                            val_iter,
                            scheduler.get_last_lr()[0],
                            torch.tensor(loss_val).mean(),
                        ))                          
        if args.iswriter:
            writer.add_scalars('loss', {"train_loss":torch.tensor(loss_train).mean(),
                                       "val_loss":torch.tensor(loss_val).mean()}, epoch)          



        save_state = {'state_dict': model.state_dict()}
        ckpt_path = os.path.join(event_path, 'checkpoint_path')


        if epoch ==already_trained_epoch:
            best_loss = torch.tensor(loss_val).mean()
        if best_loss >= torch.tensor(loss_val).mean():
            best_loss = torch.tensor(loss_val).mean()
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
        loss_val_per_epoch.append(torch.tensor(loss_val).mean());
        epoch_list.append(epoch);
        for epoch_draw_iter in epoch_list:
            if loss_train_per_epoch[epoch_draw_iter-already_trained_epoch]>100*best_loss:
                loss_train_per_epoch[epoch_draw_iter-already_trained_epoch]=0
                loss_val_per_epoch[epoch_draw_iter-already_trained_epoch]=0
        import matplotlib.pyplot as plt
        import pylab as pl
        fig=plt.figure()
        pl.plot(epoch_list,loss_train_per_epoch,"r-",label="train_loss")
        pl.plot(epoch_list,loss_val_per_epoch,"g-",label="val_loss")
        pl.legend()
        pl.xlabel("epoch")
        pl.ylabel("loss")
        plt.savefig(os.path.join(ckpt_path,"loss.jpg"))        
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
