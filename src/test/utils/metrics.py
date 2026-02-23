import torch
import torch.nn.functional as F
import torch.nn as nn
import SimpleITK as sitk
from medpy import metric
import numpy as np

# class DSC(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.smooth = 1e-6

#     def forward(self, y_pred, y_truth):
#         """
#         :param y_pred: (BS,3,336,544)
#         :param y_truth: (BS,336,544)
#         :return:
#         """
#         y_pred_f = F.one_hot(y_pred.long(), 2)  # (BS,336,544,3)
#         y_pred_f = torch.flatten(y_pred_f, start_dim=0, end_dim=2)   # (-1,3)

#         y_truth_f = F.one_hot(y_truth.long(), 2)  # (BS,336,544,3)
#         y_truth_f = torch.flatten(y_truth_f, start_dim=0, end_dim=2)  # (-1,3)

#         dice1 = (2. * ((y_pred_f[:, 1:] * y_truth_f[:, 1:]).sum()) + self.smooth) / (
#                 y_pred_f[:, 1:].sum() + y_truth_f[:, 1:].sum() + self.smooth)

#         dice1.requires_grad_(False)
#         return dice1
    
class DSC(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1e-6

    def forward(self, y_pred, y_truth):
        """
        :param y_pred: (BS,3,336,544)
        :param y_truth: (BS,336,544)
        :return:
        """
        y_pred_f = F.one_hot(y_pred.long(), 3)  # (BS,336,544,3)       
        y_pred_f = torch.flatten(y_pred_f, start_dim=0, end_dim=1)   # (-1,3)

        y_truth_f = F.one_hot(y_truth.long(), 3)  # (BS,336,544,3)
        y_truth_f = torch.flatten(y_truth_f, start_dim=0, end_dim=1)  # (-1,3)

        dice1 = (2. * ((y_pred_f[:, 1:2] * y_truth_f[:, 1:2]).sum()) + self.smooth) / (
                y_pred_f[:, 1:2].sum() + y_truth_f[:, 1:2].sum() + self.smooth)
        dice2 = (2. * ((y_pred_f[:, 2:] * y_truth_f[:, 2:]).sum()) + self.smooth) / (
                y_pred_f[:, 2:].sum() + y_truth_f[:, 2:].sum() + self.smooth)

        dice1.requires_grad_(False)
        dice2.requires_grad_(False)
        avg_dice = (dice1 + dice2) / 2
        return dice1, dice2, avg_dice


class HD(nn.Module):
    def __init__(self):
        super(HD,self).__init__()
        pass

    def numpy_to_image(self, image) -> sitk.Image:
        image = sitk.GetImageFromArray(image)
        return image

    def evaluation(self, pred: np.ndarray, label: np.ndarray):
        result = dict()

        # 计算总体指标
        pred_all = sitk.GetImageFromArray(pred)
        label_all = sitk.GetImageFromArray(label)
        
        result['hd_head'] = float(self.cal_hd(pred_all, label_all))
        
        return result['hd_head']

    def forward(self, pred, label):
        """
        :param pred: (BS,2,336,544)
        :param label: (BS,336,544)
        :return:
        """
        #print(pred.shape)
        #print(label.shape)

        pred = torch.argmax(pred,dim=1)[0].detach().cpu().numpy().astype(np.int64)  # (H,W) value:0,1  1-head
        label = label[0].detach().cpu().numpy().astype(np.int64) # (H,W) value:0,1  1-head
        pre_image = self.numpy_to_image(pred)
        truth_image = self.numpy_to_image(label)
        result = self.evaluation(pre_image, truth_image)

        return result

    def cal_hd(self, a, b):
        a = sitk.Cast(sitk.RescaleIntensity(a), sitk.sitkUInt8)
        b = sitk.Cast(sitk.RescaleIntensity(b), sitk.sitkUInt8)
        filter1 = sitk.HausdorffDistanceImageFilter()
        filter1.Execute(a, b)
        hd = filter1.GetHausdorffDistance()
        return hd
    
def calculate_metric_percase(pred, gt):
    jc = metric.binary.jc(pred, gt)
    # dc = metric.binary.dc(pred, gt)
    # hd = metric.binary.hd95(pred, gt)
    # asd = metric.binary.asd(pred, gt)

    return jc#, dc, hd, asd