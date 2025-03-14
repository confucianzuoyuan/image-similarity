__all__ = ["FolderDataset"]

import torch
from PIL import Image
import os
from tqdm import tqdm
from torch.utils.data import Dataset


class FolderDataset(Dataset):
    """
    从图像文件夹创建 PyTorch 可用的数据集，返回图像的张量表示
    
    参数:
    - main_dir : 存储图片的路径
    - transform (可选) : 用来对图像进行变换的 torchvision transforms
    """

    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")

        if self.transform is not None:
            tensor_image = self.transform(image)

        return tensor_image, tensor_image
