import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import zoom
# from torchvision import transforms
import torchvision.transforms.v2 as transforms
import albumentations as A
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import matplotlib.pyplot as plt
from PIL import Image
from torch.nn.functional import one_hot

class BaseDataSets_HC18(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        default_transform=None,
        label_transform=None,
        islabeled=True,
        isAL=False,
        alNum=2,
        stgrategy=None,
        ignoreList=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.default_transform = default_transform
        self.label_transform = label_transform
        self.islabeled = islabeled
        self.isAL = isAL
        self.alNum = alNum
        self.stgrategy = stgrategy
        self.ignoreList = ignoreList

        if not self.isAL:
            if self.split == "train" and self.islabeled:
                with open(self._base_dir + f"/ssl/train/labeled_data/data_{num}.list", "r") as f:
                    self.sample_list = f.readlines()
            elif self.split == "train" and not self.islabeled:
                if num is None:
                    with open(self._base_dir + "/ssl/train/unlabeled_data/data.list", "r") as f:
                        self.sample_list = f.readlines()
                else:
                    with open(self._base_dir + f"/ssl/train/unlabeled_data/data_{num}.list", "r") as f:
                        self.sample_list = f.readlines()
            elif self.split == "val":
                with open(self._base_dir + "/ssl/val/data.list", "r") as f:
                    self.sample_list = f.readlines()
            elif self.split == "suptrain" and num in (500,342,1084):
                with open(self._base_dir + "/ssl/train/data.list", "r") as f:
                    self.sample_list = f.readlines()
            elif self.split == "suptrain" and num not in (500,342,1084):
                with open(self._base_dir + f"/ssl/train/labeled_data/data_{num}.list", "r") as f:
                    self.sample_list = f.readlines()
        else:
            if self.split == "train" and self.islabeled:
                # with open(args.root_path + f"/ssl/train/labeled_data/{strategy_name}/data_al_{al_iter}_{labeled_slice}.list", "w") as f:
                with open(self._base_dir + f"/ssl/train/labeled_data/{self.stgrategy}/data_al_{self.alNum}_{num}.list", "r") as f1:
                    self.sample_list = f1.readlines()
            elif self.split == "train" and not self.islabeled:
                with open(self._base_dir + "/ssl/train/unlabeled_data/data.list", "r") as f1:
                    self.unlabeled_list = f1.readlines()
                with open(self._base_dir + f"/ssl/train/labeled_data/{self.stgrategy}/data_al_{self.alNum}_{num}.list", "r") as f1:
                    self.labeled_list = f1.readlines()
                self.sample_list = np.setdiff1d(self.unlabeled_list, self.labeled_list)
        
        self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        if ignoreList is not None:
            self.sample_list = list(set(self.sample_list) - set(self.ignoreList))
        
        if num is not None and self.split == "val" and not self.isAL:
            self.sample_list = self.sample_list[:num]
            
        # print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image = None
        label = None
        sample = None
        try:
            # _base_dir = '/root/autodl-tmp/HC18_semi'
            if self.islabeled:
                # print(f'labeled train case: {case}')
                image = cv2.imread(self._base_dir + "/image/{}.png".format(case), cv2.IMREAD_COLOR) #load US images
                label = cv2.imread(self._base_dir + "/mask/{}_Annotation.png".format(case), cv2.IMREAD_UNCHANGED) #load masks
                label = label/255
                label = label.astype('float32')
                # one-hot encoding of mask
                # x, y, z = image.shape
                # label = np.eye(z)[label]
            else:
                # print(f'unlabeled train case: {case}')
                image = cv2.imread(self._base_dir + "/image/{}.png".format(case), cv2.IMREAD_COLOR) #load US images
                # fake label
                x, y, z = image.shape
                label = np.zeros((x, y))
                label = label.astype('float32')
            
            if self.label_transform is None and self.transform is not None:
                # label = label*125
                augmentations = self.transform(image=image, mask=label)
                image = augmentations["image"]
                label = augmentations["mask"]
                # one-hot encoding of mask
                # print(f'after transform {label.shape}')
                # label = torch.squeeze(label)
                # label = one_hot(label.long(), 3) # num of classes = 3
                # label = label.permute(2,0,1).float()
            elif self.label_transform is not None and self.transform is not None:
                image = self.transform(image)
                label = self.label_transform(label)
            
            image = self.default_transform(image)
            label = self.default_transform(label)
            sample = {"image": image, "label": label}
            sample["name"] = case
            sample["idx"] = idx
        except Exception as e:
            print(f"message: {repr(e)}")
            print(f"An exception occurred at {case} and {self.split} and {self.islabeled}")
        return sample


class BaseDataSets_ESTT(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        default_transform=None,
        label_transform=None,
        islabeled=True,
        isAL=False,
        alNum=2,
        stgrategy=None,
        ignoreList=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.default_transform = default_transform
        self.label_transform = label_transform
        self.islabeled = islabeled
        self.isAL = isAL
        self.alNum = alNum
        self.stgrategy = stgrategy
        self.ignoreList = ignoreList

        if not self.isAL:
            if self.split == "train" and self.islabeled:
                with open(self._base_dir + f"/ssl/train/labeled_data/data_{num}.list", "r") as f:
                    self.sample_list = f.readlines()
            elif self.split == "train" and not self.islabeled:
                if num is None:
                    with open(self._base_dir + "/ssl/train/unlabeled_data/data.list", "r") as f:
                        self.sample_list = f.readlines()
                else:
                    with open(self._base_dir + f"/ssl/train/unlabeled_data/data_{num}.list", "r") as f:
                        self.sample_list = f.readlines()
            elif self.split == "val":
                with open(self._base_dir + "/ssl/val/data.list", "r") as f:
                    self.sample_list = f.readlines()
            elif self.split == "suptrain" and num in (500,342):
                with open(self._base_dir + "/ssl/train/data.list", "r") as f:
                    self.sample_list = f.readlines()
            elif self.split == "suptrain" and num not in (500,342):
                with open(self._base_dir + f"/ssl/train/labeled_data/data_{num}.list", "r") as f:
                    self.sample_list = f.readlines()
        else:
            if self.split == "train" and self.islabeled:
                # with open(args.root_path + f"/ssl/train/labeled_data/{strategy_name}/data_al_{al_iter}_{labeled_slice}.list", "w") as f:
                with open(self._base_dir + f"/ssl/train/labeled_data/{self.stgrategy}/data_al_{self.alNum}_{num}.list", "r") as f1:
                    self.sample_list = f1.readlines()
            elif self.split == "train" and not self.islabeled:
                with open(self._base_dir + "/ssl/train/unlabeled_data/data.list", "r") as f1:
                    self.unlabeled_list = f1.readlines()
                with open(self._base_dir + f"/ssl/train/labeled_data/{self.stgrategy}/data_al_{self.alNum}_{num}.list", "r") as f1:
                    self.labeled_list = f1.readlines()
                self.sample_list = np.setdiff1d(self.unlabeled_list, self.labeled_list)
        
        self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        if ignoreList is not None:
            self.sample_list = list(set(self.sample_list) - set(self.ignoreList))
        
        if num is not None and self.split == "val" and not self.isAL:
            self.sample_list = self.sample_list[:num]
            
        # print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image = None
        label = None
        sample = None
        try:
            if self.islabeled:
                image = cv2.imread(self._base_dir + "/image/{}.png".format(case), cv2.IMREAD_COLOR) #load US images
                label = cv2.imread(self._base_dir + "/mask/{}_Annotation.png".format(case), cv2.IMREAD_UNCHANGED) #load masks
                # label = label/255
                label[label != 0] = 1
                label = label.astype('float32')
            else:
                image = cv2.imread(self._base_dir + "/image/{}.png".format(case), cv2.IMREAD_COLOR) #load US images
                x, y, z = image.shape
                label = np.zeros((x, y))
                label = label.astype('float32')
            
            if self.label_transform is None and self.transform is not None:
                augmentations = self.transform(image=image, mask=label)
                image = augmentations["image"]
                label = augmentations["mask"]
            elif self.label_transform is not None and self.transform is not None:
                image = self.transform(image)
                label = self.label_transform(label)
            
            image = self.default_transform(image)
            label = self.default_transform(label)
            sample = {"image": image, "label": label}
            sample["name"] = case
            sample["idx"] = idx
        except Exception as e:
            print(f"message: {repr(e)}")
            print(f"An exception occurred at {case} and {self.split} and {self.islabeled}")
        return sample
    

class BaseDataSets_FetalAbdomen(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        default_transform=None,
        label_transform=None,
        islabeled=True,
        isAL=False,
        alNum=2,
        stgrategy=None,
        ignoreList=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.default_transform = default_transform
        self.label_transform = label_transform
        self.islabeled = islabeled
        self.isAL = isAL
        self.alNum = alNum
        self.stgrategy = stgrategy
        self.ignoreList = ignoreList

        if not self.isAL:
            if self.split == "train" and self.islabeled:
                with open(self._base_dir + f"/ssl/train/labeled_data/data_{num}.list", "r") as f:
                    self.sample_list = f.readlines()
            elif self.split == "train" and not self.islabeled:
                if num is None:
                    with open(self._base_dir + "/ssl/train/unlabeled_data/data.list", "r") as f:
                        self.sample_list = f.readlines()
                else:
                    with open(self._base_dir + f"/ssl/train/unlabeled_data/data_{num}.list", "r") as f:
                        self.sample_list = f.readlines()
            elif self.split == "val":
                with open(self._base_dir + "/ssl/val/data.list", "r") as f:
                    self.sample_list = f.readlines()
            elif self.split == "suptrain" and num in (500,342,1084):
                with open(self._base_dir + "/ssl/train/data.list", "r") as f:
                    self.sample_list = f.readlines()
            elif self.split == "suptrain" and num not in (500,342,1084):
                with open(self._base_dir + f"/ssl/train/labeled_data/data_{num}.list", "r") as f:
                    self.sample_list = f.readlines()
        else:
            if self.split == "train" and self.islabeled:
                with open(self._base_dir + f"/ssl/train/labeled_data/{self.stgrategy}/data_al_{self.alNum}_{num}.list", "r") as f1:
                    self.sample_list = f1.readlines()
            elif self.split == "train" and not self.islabeled:
                with open(self._base_dir + "/ssl/train/unlabeled_data/data.list", "r") as f1:
                    self.unlabeled_list = f1.readlines()
                with open(self._base_dir + f"/ssl/train/labeled_data/{self.stgrategy}/data_al_{self.alNum}_{num}.list", "r") as f1:
                    self.labeled_list = f1.readlines()
                self.sample_list = np.setdiff1d(self.unlabeled_list, self.labeled_list)
        
        self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        if ignoreList is not None:
            self.sample_list = list(set(self.sample_list) - set(self.ignoreList))
        
        if num is not None and self.split == "val" and not self.isAL:
            self.sample_list = self.sample_list[:num]
            
        # print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image = None
        label = None
        sample = None
        try:
            if self.islabeled:
                image = cv2.imread(self._base_dir + "/image/{}".format(case), cv2.IMREAD_COLOR) #load US images
                label = cv2.imread(self._base_dir + "/mask/{}".format(case).replace('_image_','_mask_'), cv2.IMREAD_UNCHANGED) #load masks
                # label = label/255
                label[label != 0] = 1
                label = label.astype('float32')
            else:
                image = cv2.imread(self._base_dir + "/image/{}".format(case), cv2.IMREAD_COLOR) #load US images
                x, y, z = image.shape
                label = np.zeros((x, y))
                label = label.astype('float32')
            
            if self.label_transform is None and self.transform is not None:
                augmentations = self.transform(image=image, mask=label)
                image = augmentations["image"]
                label = augmentations["mask"]
            elif self.label_transform is not None and self.transform is not None:
                image = self.transform(image)
                label = self.label_transform(label)
            
            image = self.default_transform(image)
            label = self.default_transform(label)
            sample = {"image": image, "label": label}
            sample["name"] = case
            sample["idx"] = idx
        except Exception as e:
            print(f"message: {repr(e)}")
            print(f"An exception occurred at {case} and {self.split} and {self.islabeled}")
        return sample    
    
    
class BaseDataSets_BUSI(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        transform=None,
        default_transform=None,
        label_transform=None,
        stgrategy=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.default_transform = default_transform
        self.label_transform = label_transform
        self.stgrategy = stgrategy
    
        with open(self._base_dir + f"/dataset/BUSI/{self.split}/data.list", "r") as f:
            self.sample_list = f.readlines()
        
        self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image = None
        label = None
        sample = None
        try:
            image = cv2.imread(self._base_dir + "/dataset/BUSI/image/{}".format(case), cv2.IMREAD_COLOR) #load US images
            label = cv2.imread(self._base_dir + "/dataset/BUSI/mask/{}".format(case).replace('.png','_mask.png'), cv2.IMREAD_GRAYSCALE) #load masks
            # label = label/255
            label[label != 0] = 1
            label = label.astype('float32')
            
            if self.label_transform is None and self.transform is not None:
                augmentations = self.transform(image=image, mask=label)
                image = augmentations["image"]
                label = augmentations["mask"]
            elif self.label_transform is not None and self.transform is not None:
                image = self.transform(image)
                label = self.label_transform(label)
            
            image = self.default_transform(image)
            label = self.default_transform(label)
            sample = {"image": image, "label": label}
            sample["name"] = case
            sample["idx"] = idx
        except Exception as e:
            print(f"message: {repr(e)}")
            print(f"An exception occurred at {case} and {self.split}")
        return sample
    

class BaseDataSets_PSFHS(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        default_transform=None,
        label_transform=None,
        islabeled=True,
        isAL=False,
        alNum=2,
        stgrategy=None,
        ignoreList=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.default_transform = default_transform
        self.label_transform = label_transform
        self.islabeled = islabeled
        self.isAL = isAL
        self.alNum = alNum
        self.stgrategy = stgrategy
        self.ignoreList = ignoreList

        if not self.isAL:
            if self.split == "train" and self.islabeled:
                with open(self._base_dir + f"/ssl/train/labeled_data/data_{num}.list", "r") as f:
                    self.sample_list = f.readlines()
            elif self.split == "train" and not self.islabeled:
                if num is None:
                    with open(self._base_dir + "/ssl/train/unlabeled_data/data.list", "r") as f:
                        self.sample_list = f.readlines()
                else:
                    with open(self._base_dir + f"/ssl/train/unlabeled_data/data_{num}.list", "r") as f:
                        self.sample_list = f.readlines()
            elif self.split == "val":
                with open(self._base_dir + "/ssl/val/data.list", "r") as f:
                    self.sample_list = f.readlines()
            elif self.split == "suptrain" and num in (500,342,679):
                with open(self._base_dir + "/ssl/train/data.list", "r") as f:
                    self.sample_list = f.readlines()
            elif self.split == "suptrain" and num not in (500,342,679):
                with open(self._base_dir + f"/ssl/train/labeled_data/data_{num}.list", "r") as f:
                    self.sample_list = f.readlines()
        else:
            if self.split == "train" and self.islabeled:
                # with open(args.root_path + f"/ssl/train/labeled_data/{strategy_name}/data_al_{al_iter}_{labeled_slice}.list", "w") as f:
                with open(self._base_dir + f"/ssl/train/labeled_data/{self.stgrategy}/data_al_{self.alNum}_{num}.list", "r") as f1:
                    self.sample_list = f1.readlines()
            elif self.split == "train" and not self.islabeled:
                with open(self._base_dir + "/ssl/train/unlabeled_data/data.list", "r") as f1:
                    self.unlabeled_list = f1.readlines()
                with open(self._base_dir + f"/ssl/train/labeled_data/{self.stgrategy}/data_al_{self.alNum}_{num}.list", "r") as f1:
                    self.labeled_list = f1.readlines()
                self.sample_list = np.setdiff1d(self.unlabeled_list, self.labeled_list)
        
        self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        if ignoreList is not None:
            self.sample_list = list(set(self.sample_list) - set(self.ignoreList))
        
        if num is not None and self.split == "val" and not self.isAL:
            self.sample_list = self.sample_list[:num]
            
        # print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image = None
        label = None
        sample = None
        try:
            if self.islabeled:
                image = cv2.imread(self._base_dir + "/image/{}.png".format(case), cv2.IMREAD_COLOR) #load US images
                label = cv2.imread(self._base_dir + "/mask/{}.png".format(case), cv2.IMREAD_UNCHANGED) #load masks
                # label = label/255
                # label[label != 0] = 1
                label = label.astype('float32')
            else:
                image = cv2.imread(self._base_dir + "/image/{}.png".format(case), cv2.IMREAD_COLOR) #load US images
                x, y, z = image.shape
                label = np.zeros((x, y))
                label = label.astype('float32')
            
            if self.label_transform is None and self.transform is not None:
                augmentations = self.transform(image=image, mask=label)
                image = augmentations["image"]
                label = augmentations["mask"]
            elif self.label_transform is not None and self.transform is not None:
                image = self.transform(image)
                label = self.label_transform(label)
            
            image = self.default_transform(image)
            label = self.default_transform(label)
            sample = {"image": image, "label": label}
            sample["name"] = case
            sample["idx"] = idx
        except Exception as e:
            print(f"message: {repr(e)}")
            print(f"An exception occurred at {case} and {self.split} and {self.islabeled}")
        return sample   


class BaseDataSets_Cervical(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        default_transform=None,
        label_transform=None,
        islabeled=True,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.default_transform = default_transform
        self.label_transform = label_transform
        self.islabeled = islabeled
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        if not self.isAL:
            if self.split == "train" and self.islabeled:
                with open(self._base_dir + f"/ssl/train/labeled_data/data_{num}.list", "r") as f:
                    self.sample_list = f.readlines()
            elif self.split == "train" and not self.islabeled:
                if num is None:
                    with open(self._base_dir + "/ssl/train/unlabeled_data/data.list", "r") as f:
                        self.sample_list = f.readlines()
                else:
                    with open(self._base_dir + f"/ssl/train/unlabeled_data/data_{num}.list", "r") as f:
                        self.sample_list = f.readlines()
            elif self.split == "val":
                with open(self._base_dir + "/ssl/val/data.list", "r") as f:
                    self.sample_list = f.readlines()
            elif self.split == "suptrain" and num in (500,342,679):
                with open(self._base_dir + "/ssl/train/data.list", "r") as f:
                    self.sample_list = f.readlines()
            elif self.split == "suptrain" and num not in (500,342,679):
                with open(self._base_dir + f"/ssl/train/labeled_data/data_{num}.list", "r") as f:
                    self.sample_list = f.readlines()
        else:
            if self.split == "train" and self.islabeled:
                # with open(args.root_path + f"/ssl/train/labeled_data/{strategy_name}/data_al_{al_iter}_{labeled_slice}.list", "w") as f:
                with open(self._base_dir + f"/ssl/train/labeled_data/{self.stgrategy}/data_al_{self.alNum}_{num}.list", "r") as f1:
                    self.sample_list = f1.readlines()
            elif self.split == "train" and not self.islabeled:
                with open(self._base_dir + "/ssl/train/unlabeled_data/data.list", "r") as f1:
                    self.unlabeled_list = f1.readlines()
                with open(self._base_dir + f"/ssl/train/labeled_data/{self.stgrategy}/data_al_{self.alNum}_{num}.list", "r") as f1:
                    self.labeled_list = f1.readlines()
                self.sample_list = np.setdiff1d(self.unlabeled_list, self.labeled_list)
        
        self.sample_list = [item.replace("\n", "") for item in self.sample_list]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image = None
        label = None
        sample = None
        try:
            if self.islabeled:
                # print(f'labeled train case: {case}')
                image = cv2.imread(self._base_dir + "/fetal_semi/train/labeled_data/images/{}.png".format(case), cv2.IMREAD_COLOR) #load US images
                label = cv2.imread(self._base_dir + "/fetal_semi/train/labeled_data/labels/{}.png".format(case), cv2.IMREAD_UNCHANGED) #load masks
                label = label.astype('float32')
                # one-hot encoding of mask
                # x, y, z = image.shape
                # label = np.eye(z)[label]
            else:
                # print(f'unlabeled train case: {case}')
                image = cv2.imread(self._base_dir + "/fetal_semi/train/unlabeled_data/images/{}.png".format(case), cv2.IMREAD_COLOR) #load US images
                # fake label
                x, y, z = image.shape
                label = np.zeros((x, y))
                label = label.astype('float32')
            
            if self.label_transform is None and self.transform is not None:
                # label = label*125
                augmentations = self.transform(image=image, mask=label)
                image = augmentations["image"]
                label = augmentations["mask"]
                # one-hot encoding of mask
                # print(f'after transform {label.shape}')
                # label = torch.squeeze(label)
                # label = one_hot(label.long(), 3) # num of classes = 3
                # label = label.permute(2,0,1).float()
            elif self.label_transform is not None and self.transform is not None:
                image = self.transform(image)
                label = self.label_transform(label)

            # image = torch.tensor(image).permute(2, 0, 1)
            # label = torch.tensor(label).unsqueeze(0)
            
            image = self.default_transform(image)
            label = self.default_transform(label)
            sample = {"image": image, "label": label}
            sample["name"] = case
            sample["idx"] = idx
        except Exception as e:
            print(f"message: {repr(e)}")
            print(f"An exception occurred at {case} and {self.split} and {self.islabeled}")
        return sample


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
