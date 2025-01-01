from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os


class Dataset_loader(Dataset):
    def __init__(self,args,phase="train"):
        assert phase in ["train","test","val"]
        self.phase=phase
        self.__load_data__();
        pass

    def __getitem__(self, item):
        index = item % len(self.obj_list)
        obj_name = self.obj_list[index]
        # TODO to complete，补全需要取出的数据集内容
        # eg:
        # image_path = os.path.join(obj_name ,'Image/0.png')

        # 注意要对图片类数据进行维度转换和归一化等操作
        # 其余np类数据应该会在dict_to_cuda过程中
        # 被转换为tensor
        import torchvision.transforms as transforms
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
            ]
        )
        # imA = np.array(Image.open(image0_path))
        # imA=transform(imA)
        # 附：tensor转np的代码
        # def tensor_2_image(tensor,img_save_path):
        #     array = tensor.detach().cpu().numpy().transpose((1, 2, 0))
        #     array = (array + 1) / 2.0 * 255.0
        #     array = array.astype(np.uint8)
        #     image = Image.fromarray(array)
        #     image.save(img_save_path) 

        # TODO to complete，完成数据的加载

        example_dict = {
        # TODO to complete，
        # eg:
        # "inputA": imA,
        }
        return example_dict
        pass

    def __len__(self):
        return len(self.obj_list)
        pass

    def __load_data__(self):
        # 用一个txt文件记录数据集
        # 每一行代表一条数据
        # eg：/data3/zh/utils/network_structure/template/datasets/base_data.txt
        # TODO to complete，补全数据集txt路径以及每类数据集有多少
        self.dataset_list="";
        if self.phase=="train":
            self.obj_list=[]
            with open(self.dataset_list,"r") as f:
                lines =f.readlines();
                for k in range(0,6500):
                    self.obj_list.append(lines[k].strip())
            f.close()
            pass
        elif self.phase=="val":
            self.obj_list=[]
            with open(self.dataset_list,"r") as f:
                lines =f.readlines();
                for k in range(6500,7000):
                    self.obj_list.append(lines[k].strip())
            f.close()
            pass
        elif self.phase=="test":
            self.obj_list=[]
            with open(self.dataset_list,"r") as f:
                lines =f.readlines();
                for k in range(7000,7100):
                    self.obj_list.append(lines[k].strip())
            f.close()
            pass

    pass

class Dataset_train(Dataset_loader):
    def __init__(self, args, phase="train"):
        super(Dataset_train,self).__init__(args, phase)

class Dataset_test(Dataset_loader):
    def __init__(self, args, phase="test"):
        super(Dataset_test,self).__init__(args, phase)

class Dataset_val(Dataset_loader):
    def __init__(self, args, phase="val"):
        super(Dataset_val,self).__init__(args, phase)




def main():
    
    return
if __name__ == '__main__':
    main()
