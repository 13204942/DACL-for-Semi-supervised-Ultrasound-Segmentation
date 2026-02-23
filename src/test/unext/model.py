import numpy as np
from os.path import isfile
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import cv2

from unext.archs import UNext as Unet2D
from torchvision import transforms


class model:
    def __init__(self, num_classes=2):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        '''
        # print(f'submit swin-unet unext method')
        self.mean = None
        self.std = None
        # self.model = ViM_seg(img_size=[224, 224], num_classes=3).cpu()
        # self.model = Unet2D(in_chns=3, class_num=2).cpu()
        self.model = Unet2D(in_chns=3, out_channels=num_classes, img_size=448, num_classes=num_classes).cpu()
        # self.model = U_Net().cpu()
        self.tr_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((448, 448)),
        ])

    def load(self, path="./", subfolder=None, iternum=None, ssl=True, model_name=None):
        # self.model.load_from(config)
        # model_path = os.path.join(path,"model.pth")

        model_path = ''
        
        if subfolder is not None:
            path = path + subfolder

        for f_name in os.listdir(path):
            # if f_name.startswith('model1') and f_name.endswith(f'iter_{iternum}.pth'):
            if model_name is None:
                if iternum is None and f_name.endswith('best_model1.pth'):
                    model_path = os.path.join(path,f_name)
                    print(f'found a model file {model_path}')
                elif iternum is not None and f_name.endswith(f'_{iternum}.pth'):
                    model_path = os.path.join(path,f_name)
                    print(f'found a model file {model_path}')
            else:
                if f_name == model_name:
                    model_path = os.path.join(path,f_name)
                    print(f'found a model file {model_path}')
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        return self

    def predict(self, X):
        """
        X: numpy array of shape (3,224,224)
        """
        # print('submit swin-unet unext network')
        self.model.eval()
        # X = X / 255.0
        # print(f'{X.shape}')
        # print(f'X shape {X.shape}')

        # X = np.transpose(X, axes=[1,2,0])

        image = self.tr_transforms(X) # image (3,448,448)
        image = image.unsqueeze(0) # image (1,3,448,448)    

        seg, _ = self.model(image)  # seg (1,3,448,448)
        seg = torch.softmax(seg, dim=1) # seg (1,3,448,448)

        # seg = seg.argmax(dim=0).detach().numpy()  # (448,448) values:{0,1,2} 1 upper 2 lower
        seg = seg.squeeze(0).argmax(dim=0).detach().numpy()  # (448,448) values:{0,1,2} 1 upper 2 lower
        # seg = cv2.resize(seg, (544,336), 0, 0, interpolation = cv2.INTER_NEAREST)
        seg = seg.astype(np.uint8)

        return seg

    def save(self, path="./"):
        '''
        Save a trained model.
        '''
        pass
