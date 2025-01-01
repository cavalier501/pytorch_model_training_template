import os
import numpy as np
import torch
import torch.nn as nn
import os
import time
from PIL import Image
import pickle


def listFiles(dirPath,txt_path):
    # 准备一个空列表,用来存储遍历数据
    f=open(txt_path, 'w');
    # fileList = []
    
    ''' os.walk(dirPath) :走查指定的文件夹路径
            root  :代表目录的路径
            dirs  :一个list,包含了dirpath下所有子目录文件夹的名字
            files :一个list,包含了所有非目录文件的名字
    '''
    i=0;
    for root, dirs, files in os.walk(dirPath):
    
        # 循环遍历列表:files【所有文件】,仅得到不包含路径的文件名
        for file in files:
        
            # 空列表写入遍历的文件名称,并用目录路径拼接文件名称
            # fileList.append(os.path.join(root, fileObj))
            # print(file)
            if i%2==0:
                f.write(os.path.join(root, file)+"\n")
            i+=1;
            
    # 打印一下列表存储内容:指定文件夹下所有的文件名
    

def listFiles_test():
    # dirPath="/nfsdata/datasets/Face/facescape_data256/Depth_noise_sigma=0.001";
    dirPath="/mnt/smbdata/Face/detail_faceScape"
    txt_path="./out.txt"
    listFiles(dirPath,txt_path);
    return


def get_facescape_data_raw():
    id_lists=np.arange(1,320+1).tolist()
    id_lists=[str(num) for num in id_lists]
    exp_lists = ['1_neutral', '2_smile', '3_mouth_stretch', '4_anger', '5_jaw_left', \
                         '6_jaw_right', '7_jaw_forward', '8_mouth_left', '9_mouth_right', '10_dimpler', \
                         '11_chin_raiser', '12_lip_puckerer', '13_lip_funneler', '14_sadness', '15_lip_roll', \
                         '16_grin', '17_cheek_blowing', '18_eye_closed', '19_brow_raiser', '20_brow_lower']            
    good_data_file="./test/get_dataset/good_data.txt"
    bad_data_file="./test/get_dataset/bad_data.txt"
    good_data_list=[]
    bad_data_list=[]
    root="/mnt/smbdata/Face/FaceScapeRawData/fsmview_trainset_shape"
    for id_iter in range(len(id_lists)):
        for exp_iter in range(len(exp_lists)):
            root_complete=1
            if root_complete:
                if id_iter+1 >= 1 and id_iter+1 <= 20:
                    root_ = os.path.join(root, "fsmview_trainset_shape_001-020")
                elif id_iter+1 >= 21 and id_iter+1 <= 40:
                    root_ = os.path.join(root, "fsmview_trainset_shape_021-040")
                elif id_iter+1 >= 41 and id_iter+1 <= 60:
                    root_ = os.path.join(root, "fsmview_trainset_shape_041-060")
                elif id_iter+1 >= 61 and id_iter+1 <= 80:
                    root_ = os.path.join(root, "fsmview_trainset_shape_061-080")
                elif id_iter+1 >= 81 and id_iter+1 <= 100:
                    root_ = os.path.join(root, "fsmview_trainset_shape_081-100")
                elif id_iter+1 >= 101 and id_iter+1 <= 120:
                    root_ = os.path.join(root, "fsmview_trainset_shape_101-120")
                elif id_iter+1 >= 121 and id_iter+1 <= 140:
                    root_ = os.path.join(root, "fsmview_trainset_shape_121-140")
                elif id_iter+1 >= 141 and id_iter+1 <= 160:
                    root_ = os.path.join(root, "fsmview_trainset_shape_141-160")
                elif id_iter+1 >= 161 and id_iter+1 <= 180:
                    root_ = os.path.join(root, "fsmview_trainset_shape_161-180")
                elif id_iter+1 >= 181 and id_iter+1 <= 200:
                    root_ = os.path.join(root, "fsmview_trainset_shape_181-200")
                elif id_iter+1 >= 201 and id_iter+1 <= 220:
                    root_ = os.path.join(root, "fsmview_trainset_shape_201-220")
                elif id_iter+1 >= 221 and id_iter+1 <= 240:
                    root_ = os.path.join(root, "fsmview_trainset_shape_221-240")
                elif id_iter+1 >= 241 and id_iter+1 <= 260:
                    root_ = os.path.join(root, "fsmview_trainset_shape_241-260")
                elif id_iter+1 >= 261 and id_iter+1 <= 280:
                    root_ = os.path.join(root, "fsmview_trainset_shape_261-280")
                elif id_iter+1 >= 281 and id_iter+1 <= 300:
                    root_ = os.path.join(root, "fsmview_trainset_shape_281-300")
                elif id_iter+1 >= 301 and id_iter+1 <= 320:
                    root_ = os.path.join(root, "fsmview_trainset_shape_301-320")
            item_name=os.path.join(root_,id_lists[id_iter],exp_lists[exp_iter]+".ply")
            
            if os.path.exists(item_name):
                good_data_list.append(item_name)
            else:
                print(item_name)
                bad_data_list.append(item_name)
    with open(good_data_file,"w") as f:
        for s in good_data_list:
            f.write(s+"\n")
    with open(bad_data_file,"w") as f:
        for s in bad_data_list:
            print(f"aaa,{s}")
            f.write(s+"\n")
    print(bad_data_list)
    return

def get_facescape_data():
    id_lists=np.arange(1,320+1).tolist()
    id_lists=[str(num) for num in id_lists]
    exp_lists = ['1_neutral', '2_smile', '3_mouth_stretch', '4_anger', '5_jaw_left', \
                         '6_jaw_right', '7_jaw_forward', '8_mouth_left', '9_mouth_right', '10_dimpler', \
                         '11_chin_raiser', '12_lip_puckerer', '13_lip_funneler', '14_sadness', '15_lip_roll', \
                         '16_grin', '17_cheek_blowing', '18_eye_closed', '19_brow_raiser', '20_brow_lower']            
    good_data_file="./test/get_dataset/good_data.txt"
    bad_data_file="./test/get_dataset/bad_data.txt"
    good_data_list=[]
    bad_data_list=[]
    root="/mnt/smbdata/Face/detailGAN"
    for id_iter in range(len(id_lists)):
        for exp_iter in range(len(exp_lists)):
            item_name=os.path.join(root,id_lists[id_iter],exp_lists[exp_iter],"detail_pts.npy")
            
            if os.path.exists(item_name):
                good_data_list.append(item_name)
            else:
                # print(item_name)
                bad_data_list.append(item_name)
    with open(good_data_file,"w") as f:
        for s in good_data_list:
            f.write(s+"\n")
    with open(bad_data_file,"w") as f:
        for s in bad_data_list:
            f.write(s+"\n")
    return


def main():
    # get_facescape_data()

    return
if __name__ == '__main__':
    main()
