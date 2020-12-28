from torch.utils.data import Dataset, DataLoader, random_split
import cv2
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
# from utils import is_image_file, load_img
import os
import numpy as np
import cv2
import argparse
import glob
from PIL import Image
from torchvision import transforms as T
from torchvision import transforms
import IPython.display as display
import torch.utils.data as data
import torch
import numpy as np
from skimage.transform import pyramid_gaussian, resize
# teacher
import numpy as np
from skimage.transform import pyramid_gaussian
import cv2
from scipy import signal
import sys
# from cv2.ximgproc import guidedFilter
import random
from skimage.color import rgb2ycbcr
from skimage.transform import pyramid_gaussian, resize
from skimage.io import imsave
from skimage import img_as_ubyte


class HDRdatasets_dynamic_compose(data.Dataset):

    def __init__(self, train=True, transforms=None):
        out_img_train = []
        gt_img_train = []
        img = []
        if train:
            for i in os.listdir(r'C:\Users\admin\Downloads\random_select'):
                img_list = glob.glob(
                    r'C:\Users\admin\Downloads\random_select' + '\\' + i + r'\*.JPG')

                for j in img_list:
                    if 'GT.JPG' in j:
                        gt_img_train.append(j)
                    else:
                        img.append(j)

        else:
            for i in os.listdir(r'C:\Users\admin\Downloads\random_select2'):
                img_list = glob.glob(
                    r'C:\Users\admin\Downloads\random_select2' + '\\' + i + r'\*.JPG')
                img_list1 = glob.glob(
                    r'C:\Users\admin\Downloads\random_select2' + '\\' + i + r'\*.png')
                for j in img_list:
                    if 'GT.JPG' in j:
                        gt_img_train.append(j)
                    else:
                        img.append(j)
        self.train = train
        self.gt_img_train = gt_img_train
        self.out_img_train = out_img_train
        self.img = img
        self.transforms = transforms


    def __getitem__(self, index):

        augmentation = False
        filename = self.img[2 * index]
        label_path = self.gt_img_train[index]
        img1_path = self.img[2 * index]
        img2_path = self.img[2 * index + 1]
        label = Image.open(label_path).convert('YCbCr').resize((512, 512))
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        img1_ycbcr = img1.convert('YCbCr')
        img2_ycbcr = img2.convert('YCbCr')
        img1_np = np.array(img1.resize((512, 512)))
        img2_np = np.array(img2.resize((512, 512)))

        if augmentation:
            img1_np = adjust_gamma(img1_np, gamma=random.uniform(0.5, 4))
            img2_np = adjust_gamma(img2_np, gamma=random.uniform(0.5, 4))

        if self.transforms:
            img1 = self.transforms(img1_ycbcr)
            img2 = self.transforms(img2_ycbcr)
            label = self.transforms(label)
        return img1, img2,  label, filename

    def __len__(self):

        return len(self.gt_img_train)




def get_loader(root, batch_size, shuffle=True):
    transforms = T.Compose([T.Resize([512, 512]), T.ToTensor(
    ), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = HDRdatasets_dynamic_compose(True, transforms)
    test_dataset = HDRdatasets_dynamic_compose(False, transforms)
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    return train_dataloader, test_dataloader
