import numpy as np
from os.path import isfile
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
try:
    import cv2
except:
    print(f'cannot import cv2')
    pass
# import cv2

from swinunet.vision_transformer import SwinUnet as ViM_seg
from torchvision import transforms


class model:
    def __init__(self, num_classes=2):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        '''
        # print(f'submit swin swin networks')
        self.mean = None
        self.std = None
        self.model = ViM_seg(img_size=[448, 448], num_classes=num_classes).cpu()
        # self.model = U_Net().cpu()
        self.tr_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((448, 448)),
        ])

        self.out_transforms = transforms.Compose([
            transforms.Resize((336, 544)),
        ])
        self.model_path = ''

    def load(self, path="./", iternum=None, ssl=True, model_name=None):
        # self.model.load_from(config)
        # model_path = os.path.join(path,"model.pth")

        # model_path = ''
        for f_name in os.listdir(path):
            # if f_name.startswith('model1') and f_name.endswith(f'iter_{iternum}.pth'):
            if model_name is None:
                if iternum is None and f_name.endswith('best_model1.pth'):
                    self.model_path = os.path.join(path,f_name)
                    print(f'found a model file {self.model_path}')
                elif iternum is not None and f_name.endswith(f'iter_{iternum}.pth'):
                    self.model_path = os.path.join(path,f_name)
                    print(f'found a model file {self.model_path}')
            else:
                if f_name == model_name:
                    self.model_path = os.path.join(path,f_name)
                    print(f'found a model file {self.model_path}')
        self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        return self

    def predict(self, X):
        """
        X: numpy array of shape (3,336,544)
        """
        # print('submit swin swin network')
        self.model.eval()
        # X = X / 255.0
        # print(f'{X.shape}')
        # print(f'X shape {X.shape}')

        # X = np.transpose(X, axes=[1,2,0])

        image = self.tr_transforms(X) # image (3,224,224)
        image = image.unsqueeze(0) # image (1,3,224,224)
        # print(f'tr_transforms image shape {image.shape}')
        # print(f'tr_transforms image max {image.max()}')
        # print(f'tr_transforms image min {image.min()}')

        seg, _ = self.model(image)  # seg (1,3,224,224)
        seg = torch.softmax(seg, dim=1) # seg (1,3,224,224)
        # seg = self.out_transforms(seg) # seg (1,3,336,544)

        seg = seg.squeeze(0).argmax(dim=0).detach().numpy()  # (224,224) values:{0,1,2} 1 upper 2 lower
        # seg = cv2.resize(seg, (544,336), 0, 0, interpolation = cv2.INTER_NEAREST)
        seg = seg.astype(np.uint8)
        # print(f'seg shape {seg.shape}')
        # print(f'seg max {seg.max()}')
        # print(f'seg min {seg.min()}')

        return seg

    def save(self, path="./"):
        '''
        Save a trained model.
        '''
        pass
