import argparse
import os
from PIL import Image
import torchvision.transforms as transforms
import time
from configuration import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
from model.model import My_model
import losses
import logging



def inference(args):
    os.environ['CUDA_VISIBLE_DEVICES']=args.CUDA_VISIBLE_DEVICES
    os.environ['CUDA_LAUNCH_BLOCKING']=args.CUDA_VISIBLE_DEVICES
    args.state ='test';
    event_path=make_dirs(args);
    logger = configure_logging(event_path)

    # TODO load the network
    model=My_model().cuda();
    ckpt = torch.load('./experiment/train/.../checkpoint_path/best_encoder.pth.tar');
    model.load_state_dict(ckpt["..."]);
    model.eval();

    # TODO 加载一些测试辅助工具
    # eg:
    Tri = np.load("/data3/zh/demo_code/model/facesacpe/predefine_data/front_faces.npy");
    core_tensor_np=np.load("/data3/zh/demo_code/model/Detailed3DFace/predef/core_847_50_52.npy");
    core_tensor=torch.from_numpy(core_tensor_np).requires_grad_(False);
    core_tensor = core_tensor.permute(2, 1, 0);
    for i in range(51):
        core_tensor[:, i + 1, :] = core_tensor[:, i + 1, :] - core_tensor[:, 0, :];
    core_tensor=core_tensor.cuda();

    # TODO load one data
    def load_one_data():
        obj_name="/nfsdata/datasets/Face/detailGAN/1/1_neutral"
        image_0_path = os.path.join(obj_name ,'Image_256/0.png')
        image_1_path = os.path.join(obj_name ,'Image_256/1.png')
        camera_0_path= os.path.join(obj_name ,'Image_256/camera0.npz')
        camera_1_path= os.path.join(obj_name ,'Image_256/camera1.npz')
        detail_pts_path=os.path.join(obj_name,"detail_pts.npy")
        
        import torchvision.transforms as transforms
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ]
        )        
        img_0=transform(np.array(Image.open(image_0_path)))
        img_1=transform(np.array(Image.open(image_1_path)))
        detail_pts=np.load(detail_pts_path)
        camera_0=np.load(camera_0_path)
        camera_1=np.load(camera_1_path)
        K_0=camera_0["K"];R_0=camera_0["R"];T_0=camera_0["T"]
        K_inverse_0=np.linalg.inv(K_0)
        R_inverse_0=np.linalg.inv(R_0)
        K_1=camera_1["K"];R_1=camera_1["R"];T_1=camera_1["T"]
        K_inverse_1=np.linalg.inv(K_1)
        R_inverse_1=np.linalg.inv(R_1)        
        example_dict={
            "inputA":img_0,
            "inputB":img_1,
            "K_inverse_A":K_inverse_0,
            "K_inverse_B":K_inverse_1,
            "R_inverse_A":R_inverse_0,
            "R_inverse_B":R_inverse_1,
            "T_A":T_0,
            "T_B":T_1,
            "detail_pts":detail_pts.reshape(-1,1),
            "basename":obj_name
        }
        return example_dict

    # TODO inference
    input_dict=load_one_data()
    input_dict = dict_to_cuda(input_dict);
    img_A=input_dict["inputA"].cuda().float().unsqueeze(0);
    img_B=input_dict["inputB"].cuda().float().unsqueeze(0);
    K_inverse_A=input_dict["K_inverse_A"].cuda().float().unsqueeze(0)
    K_inverse_B=input_dict["K_inverse_B"].cuda().float().unsqueeze(0)
    R_inverse_A=input_dict["R_inverse_A"].cuda().float().unsqueeze(0)
    R_inverse_B=input_dict["R_inverse_B"].cuda().float().unsqueeze(0)
    T_A=input_dict["T_A"].cuda().float().unsqueeze(0)
    T_B=input_dict["T_B"].cuda().float().unsqueeze(0)
    output_para=model.forward(
        imga=img_A,imgb=img_B,
        K_inverse_a=K_inverse_A,K_inverse_b=K_inverse_B,
        R_inverse_a=R_inverse_A,R_inverse_b=R_inverse_B,
        T_a=T_A,T_b=T_B
    )
    id_para=output_para[:,0:50];exp_para=output_para[:,50:102]

    # TODO save result if necessary
    save_result=1
    if save_result:
        save_result_path=os.path.join(event_path, 'checkpoint_path')
        pass

    return
def main(args):
    inference(args)
    return
if __name__ == '__main__':
    args.CUDA_VISIBLE_DEVICES="3"
    main(args)
