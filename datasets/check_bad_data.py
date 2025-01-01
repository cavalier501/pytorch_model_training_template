# 用于逐行检查每一行的数据是否有错误
import numpy as np
import torch
import torch.nn as nn
import os
import time
from PIL import Image



def check_bad_data():
    dataset_list="/data3/zh/mvfr/tu_based_mvfr/mvfnet_2/datasets/base_data.txt"
    with open(dataset_list,"r") as f:
        lines =f.readlines();
    for line in lines:
        line=line.strip()
        image_0_path = os.path.join(line ,'Image_256/0.png')
        image_1_path = os.path.join(line ,'Image_256/1.png')
        image_2_path = os.path.join(line ,'Image_256/2.png')
        camera_0_path= os.path.join(line ,'Image_256/camera0.npz')
        camera_1_path= os.path.join(line ,'Image_256/camera1.npz')
        camera_2_path= os.path.join(line ,'Image_256/camera2.npz')

        if os.path.exists(image_0_path)==False:
            # print(image_0_path)
            print(f"image_0_path not exis {line}")
            continue
        elif os.path.exists(image_1_path)==False:
            print(f"image_1_path not exis {line}")
            continue
        elif os.path.exists(image_2_path)==False:
            print(f"image_2_path not exis {line}")
            continue
        elif os.path.exists(camera_0_path)==False:
            print(f"camera_0_path not exis {line}")
            continue
        elif os.path.exists(camera_1_path)==False:
            print(f"camera_1_path not exis {line}")
            continue
        elif os.path.exists(camera_2_path)==False:
            print(f"camera_2_path not exis {line}")
            continue                
    return
def main():
    check_bad_data()
    return
if __name__ == '__main__':
    main()
