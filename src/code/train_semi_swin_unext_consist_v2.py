# autodl-tmp/Mamba-UNet/code/train_fully_supervised_2D_ViT.py
import argparse
import logging
import os
import random
import shutil
import sys
import time
import cv2

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms
import albumentations as A
from tqdm import tqdm
from networks.vision_transformer import SwinUnet as ViM_seg
from config import get_config
from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler, WeakStrongAugment)
# from dataloaders.dataset_fetalhead import BaseDataSets_HC18 as USDataSets
# from dataloaders.dataset_fetalhead import BaseDataSets_ESTT as USDataSets
# from dataloaders.dataset_fetalhead import BaseDataSets_PSFHS as USDataSets
from dataloaders.dataset_fetalhead import BaseDataSets_FetalAbdomen as USDataSets
# from dataloaders.dataset_fetalhead import BaseDataSets_Cervical as USDataSets
# from dataloaders.dataset_fetalhead import BaseDataSets_BUSI as USDataSets
from networks.vision_transformer import SwinUnet as ViT_seg
from networks.archs import UNext as Unet2D
from networks.projector import projectors as proj
from networks.net_factory import net_factory
from utils import losses, ramps #, metrics
from utils.dice import DSC
from utils.metrics import dice_score, dice
from utils.losses import ConstraLoss, ConstraLoss_AvgProj, info_nce_loss, hd_loss, softmax_kl_loss, softmax_mse_loss, ConstraLoss_multi_AvgProj, smooth_l1_loss, global_contra_loss, Uncertainty_Loss


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Semi_Mamba_UNet', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='mambaunet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[448, 448],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1234, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument(
    # '--cfg', type=str, default="../code/configs/vmamba_tiny.yaml", help='path to config file', )
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )

parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=1.0, help='consistency')
parser.add_argument('--contrastive', type=float,
                    default=1.0, help='contrastive consistency')
parser.add_argument('--cps', type=float,
                    default=1.0, help='cps')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--ict_alpha', type=float, default=0.5,
                    help='ict_alpha')
args = parser.parse_args()
config = get_config(args)
# print(config)

device_name = "cuda:0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def get_current_consistency_weight(epoch, weight):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return weight * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def discrepancy_calc_2(v1, v2):
    n, c, h, w = v1.size()    
    kl = torch.sum(v1 * torch.log((v1 + 1e-8) / (v2 + 1e-8)), dim=1)  # [B, H, W]
    # print(f'kl shape: {kl.shape}')
    # cos = F.cosine_similarity(v1, v2, dim=1)  # [B, H, W]
    # cos_dis = 1 - cos  # Higher means more disagreement

    # # Normalize both to similar range (0..1)
    # cos_dis_norm = cos_dis / cos_dis.max().clamp(min=1e-8)
    # kl_norm = kl / kl.max().clamp(min=1e-8)
    # cos_dis_norm = F.normalize(cos_dis)
    # kl_norm = F.normalize(kl)
    # print(cos_dis_norm.shape)
    # print(kl_norm.shape)
    # # loss = cos_dis_norm.mean() - kl_norm.mean()
    # # loss = torch.sum(cos_dis_norm) - torch.sum(kl_norm)
    # loss = torch.sqrt((torch.sum(cos_dis_norm) - torch.sum(kl_norm)) ** 2)
    # loss = loss / (h * w)

    # Mutual Information Gap
    H_p1 = -torch.sum(v1 * torch.log(v1 + 1e-8), dim=1)  # entropy
    H_p2 = -torch.sum(v2 * torch.log(v2 + 1e-8), dim=1)
    H_joint = -torch.sum(0.5*(v1+v2) * torch.log(0.5*(v1+v2) + 1e-8), dim=1)
    loss = (H_p1 + H_p2 - H_joint).mean() + kl.mean()
    # print(loss.shape)
    # print(f'discrepancy_calc_2: {loss}')
    return loss


def discrepancy_calc(v1, v2):
    """
    dis_loss for two different classifiers
    input : v1,v2
    output : discrepancy
    """
    assert v1.dim() == 4 # [B, C, H, W]
    assert v2.dim() == 4 # [B, C, H, W]
    n, c, h, w = v1.size()
    inner = torch.mul(v1, v2)
    print(inner.shape)
    v1 = v1.permute(2, 3, 1, 0)
    v2 = v2.permute(2, 3, 0, 1)
    mul = v1.matmul(v2)
    mul = mul.permute(2, 3, 0, 1)
    print(mul.shape)
    dis = torch.sum(mul) - torch.sum(inner)
    print(torch.sum(mul))
    print(torch.sum(inner))
    print(dis)
    dis = dis / (h * w)
    print(f'discrepancy_calc: {dis}')
    return dis


def sharpening(P):
    T = 1/args.temperature
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    labeled_slice = args.labeled_num
    val_size = 50

    con_rampup = args.consistency_rampup
    # iter_num//100
    rampup_rate =  int((labeled_slice / args.labeled_bs) * 400 / con_rampup)
    
    model1 = Unet2D(in_chns=3,
                   out_channels=args.num_classes,
                   img_size=args.patch_size[0],
                   num_classes=args.num_classes).to(device)
    
    # model2 = Unet2D(in_chns=3,
    #                out_channels=args.num_classes,
    #                img_size=args.patch_size[0],
    #                num_classes=args.num_classes).to(device)

    model2 = ViT_seg(config, 
                     img_size=args.patch_size[0],
                     num_classes=args.num_classes).to(device)
    model2.load_from(config, device_name)
    # print(model2)
    # print(type(config))
    # print(config)

    # ema_model = Unet2D(in_chns=3,
    #                out_channels=args.num_classes,
    #                img_size=args.patch_size[0],
    #                num_classes=args.num_classes).to(device)

    ema_model = ViT_seg(config, 
                        img_size=args.patch_size[0],
                        num_classes=args.num_classes).to(device)
    ema_model.load_from(config, device_name)

    for param in ema_model.parameters():
        param.detach_()    
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    
    tr_transforms = A.Compose(
        [
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2),contrast_limit=(-0.5, 0.5)),
            A.Blur(blur_limit=(3, 3), p=0.3),
            A.GaussNoise(std_range=(0.05, 0.1), p=0.3),
         ],
    )
    
    tensor_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.patch_size[0], args.patch_size[1])),
    ])  
    
    db_train_labled = USDataSets(base_dir=args.root_path, split="train", num=labeled_slice, islabeled=True, 
                                         transform=tr_transforms, default_transform=tensor_transforms) 
    
    db_train_unlabled = USDataSets(base_dir=args.root_path, split="train", num=labeled_slice, islabeled=False, 
                                           transform=None, default_transform=tensor_transforms)
    
    db_train = ConcatDataset([db_train_labled, db_train_unlabled])
    logging.info(f'total lenght of train data: {len(db_train)}')
    
    db_val = USDataSets(base_dir=args.root_path, split="val", num=val_size, islabeled=True,
                        transform=None, default_transform=tensor_transforms)
    print(len(db_val))
    # samples = random.sample(range(1, len(db_val)), val_size)
    # db_val = Subset(db_val, samples)

    logging.info("Val silices is: {}".format(len(db_val)))
    total_slices = len(db_train)
    logging.info("Total silices is: {}, labeled slices is: {}".format(total_slices, len(db_train_labled)))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)
    
    # print("labeled_idxs: {}, unlabeled_idxs: {}".format(labeled_idxs, unlabeled_idxs)) 

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    
    model1.train()
    model2.train()
    
    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    time1 = int(time.time())
    writer = SummaryWriter(snapshot_path + f'/log_{time1}')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    evalue = DSC()  # metric to find best model
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
            labeled_volume_batch = volume_batch[:args.labeled_bs]
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]
            # noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            # ema_inputs = unlabeled_volume_batch + noise
            
            outputs1, _ = model1(volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)
            
            outputs2, _ = model2(volume_batch)  # B, 14*14, 768
            outputs_soft2 = torch.softmax(outputs2, dim=1)

            # ICT mix factors
            unlabeled_bs = batch_size-args.labeled_bs
            ict_mix_factors = np.random.beta(args.ict_alpha, args.ict_alpha, size=(unlabeled_bs//2, 1, 1, 1))
            ict_mix_factors = torch.tensor(ict_mix_factors, dtype=torch.float).to(device)
            unlabeled_volume_batch_0 = unlabeled_volume_batch[0:unlabeled_bs//2, ...]
            unlabeled_volume_batch_1 = unlabeled_volume_batch[unlabeled_bs//2:, ...]

            # Mix images
            batch_ux_mixed = unlabeled_volume_batch_0 * (1.0 - ict_mix_factors) + unlabeled_volume_batch_1 * ict_mix_factors
            # input_volume_batch = torch.cat([labeled_volume_batch, batch_ux_mixed], dim=0)
            # mix_outputs1, _ = model1(batch_ux_mixed)
            mix_outputs2, _ = model2(batch_ux_mixed)
            # mix_outputs_soft1 = torch.softmax(mix_outputs1, dim=1)
            mix_outputs_soft2 = torch.softmax(mix_outputs2, dim=1)
            with torch.no_grad():
                ema_unlabeled_0, _ = ema_model(unlabeled_volume_batch_0)
                ema_unlabeled_1, _ = ema_model(unlabeled_volume_batch_1)
                ema_output_ux0 = torch.softmax(ema_unlabeled_0, dim=1)
                ema_output_ux1 = torch.softmax(ema_unlabeled_1, dim=1)
                batch_pred_mixed = ema_output_ux0 * (1.0 - ict_mix_factors) + ema_output_ux1 * ict_mix_factors
            
            # consistency_loss = torch.mean((outputs_soft - batch_pred_mixed) ** 2)
            if iter_num < 1000:
                # consistency_loss1 = 0.0
                consistency_loss2 = 0.0
            else:
                # consistency_loss1 = torch.mean((mix_outputs_soft1 - batch_pred_mixed) ** 2)
                consistency_loss2 = torch.mean((mix_outputs_soft2 - batch_pred_mixed) ** 2)
            # consistency_loss1 = Uncertainty_Loss(mix_outputs_soft1, batch_pred_mixed.detach())
            # consistency_loss2 = Uncertainty_Loss(mix_outputs_soft2, batch_pred_mixed.detach())

            # ema_output, _ = ema_model(ema_inputs)
            # consistency_loss = torch.mean((outputs1[args.labeled_bs:] - ema_output_soft)**2)

            # with torch.no_grad():
            #     ema_output, _ = ema_model(ema_inputs)
            #     ema_output_soft = torch.softmax(ema_output, dim=1)

            # ema_outputs1, _ = model1(ema_inputs)
            # ema_outputs2, _ = model2(ema_inputs)
            # consistency_loss = torch.mean(softmax_mse_loss(outputs1[args.labeled_bs:], ema_outputs2.detach()))

            # print(f'label_batch shape: {label_batch.shape}')
            # print(f'label_batch max and min: {label_batch.max()}, {label_batch.min()}')
            # print(f'outputs1 shape: {outputs1.shape}, outputs2 shape: {outputs2.shape}')
            # print(f'outputs1 max and min: {outputs1.max()}, {outputs1.min()}')
            # print(f'outputs_soft1 shape: {outputs_soft1.shape}, outputs_soft2 shape: {outputs_soft2.shape}')
            # print(f'outputs_soft1 max and min: {outputs_soft1.max()}, {outputs_soft1.min()}')

            # Ours
            # print(f'outputs1 shape: {outputs1.shape}, outputs2 shape: {ema_outputs2[:,1:,:,:].shape}')
            ####### BEST
            # consistency_loss = ConstraLoss_AvgProj(outputs1[args.labeled_bs:,1:,:,:], 
            #                                        ema_outputs2[:,1:,:,:].detach(),
            #                                        ndf=1)  # 1, 2, 4, 8, 16, 32, 64, 128

            # consistency_loss = ConstraLoss_multi_AvgProj(outputs1[args.labeled_bs:,1:,:,:], 
            #                                        ema_outputs2[:,1:,:,:].detach(),
            #                                        ndf=128)  # 1, 2, 4, 8, 16, 32, 64, 128
            # print(consistency_loss)

            # p = outputs1[args.labeled_bs:]
            # q = ema_outputs2[:,:,:,:]
            # print(f'p shape: {p.shape}, q shape: {q.shape}')
            # print(f'p max: {torch.max(p)}, p min: {torch.min(p)}')
            # print(f'q max: {torch.max(q)}, q min: {torch.min(q)}')
            # new_loss = softmax_kl_loss(p, q.detach(), sigmoid=False) # ignore background prediction
            # new_loss = smooth_l1_loss(outputs1[args.labeled_bs:,1:,:,:], outputs2[args.labeled_bs:,1:,:,:].detach()) # ignore background prediction

            ####### BEST
            # consistency_loss1 = ConstraLoss(ema_outputs1, ema_outputs2)
            # new_loss = global_contra_loss(outputs1[args.labeled_bs:,1:,:,:].detach(), ema_outputs2[:,1:,:,:])
            # print(new_loss)

            # consistency_weight = get_current_consistency_weight(iter_num // rampup_rate)
            consistency_weight_cps = get_current_consistency_weight(iter_num // rampup_rate, args.cps)
            consistency_weight_mt = get_current_consistency_weight(iter_num // rampup_rate, args.consistency)
            contra_weight_contra = get_current_consistency_weight(iter_num // rampup_rate, args.contrastive)

            # if iter_num < 1000:
            #     consistency_loss2 = 0.0
            #     consistency_loss1 = 0.0
            # else:
                # consistency_loss1 = softmax_mse_loss(outputs1[args.labeled_bs:], ema_output_soft.detach())
                # consistency_loss2 = softmax_mse_loss(outputs2[args.labeled_bs:], ema_output_soft.detach())
            
            loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs], torch.squeeze(label_batch[:args.labeled_bs],1).long()) + 
                           dice_loss(outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs]))
            
            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs], torch.squeeze(label_batch[:args.labeled_bs],1).long()) + 
                           dice_loss(outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs]))
            
            pseudo_outputs1 = torch.argmax(outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)

            pseudo_supervision1 = dice_loss(outputs_soft1[args.labeled_bs:], pseudo_outputs2.unsqueeze(1))
            pseudo_supervision2 = dice_loss(outputs_soft2[args.labeled_bs:], pseudo_outputs1.unsqueeze(1))
            # print(pseudo_supervision1)
            
            # contra = ConstraLoss_AvgProj(outputs1, outputs2, ndf=1)
            target_pred1 = F.softmax(outputs1)
            target_pred2 = F.softmax(outputs2)
            # l_cdd1 = discrepancy_calc(target_pred1, target_pred2)
            l_cdd = discrepancy_calc_2(target_pred1, target_pred2)

            # only L
            # kl_loss = softmax_kl_loss(outputs1[0,1,:,:], outputs2[0,1,:,:], sigmoid=True) # ignore background prediction
            # model1_loss = loss1 + consistency_weight * pseudo_supervision1 + 0.5*con1 #+ 0.5*kl_loss
            # model2_loss = loss2 + consistency_weight * pseudo_supervision2 + 0.5*con1 #+ 0.5*kl_loss

            model1_loss = loss1 + consistency_weight_cps * pseudo_supervision1 #+ consistency_weight_mt * consistency_loss1
            model2_loss = loss2 + consistency_weight_cps * pseudo_supervision2 + consistency_weight_mt * consistency_loss2 # ablation study

            # model1_loss = loss1 + consistency_weight * pseudo_supervision1 + 0.5*consistency_loss
            # model2_loss = loss2 + consistency_weight * pseudo_supervision2 + 0.5*consistency_loss

            # model1_loss = loss1 + consistency_weight * pseudo_supervision1 + 0.5*new_loss
            # model2_loss = loss2 + consistency_weight * pseudo_supervision2 + 0.5*new_loss

            loss = model1_loss + model2_loss + contra_weight_contra * l_cdd #+ consistency_weight_mt * consistency_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            update_ema_variables(model2, ema_model, args.ema_decay, iter_num)

            # update_ema_variables(model1, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight_cps, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            logging.info('iteration %d : model1 loss : %f model2 loss : %f' % (
                iter_num, model1_loss.item(), model2_loss.item()))
            
            if iter_num > 0 and iter_num % (labeled_slice / args.labeled_bs) == 0:
                model1.eval()
                model2.eval()
                
                metric_list1 = 0.0
                metric_list2 = 0.0
                
                for i_batch, sampled_batch in enumerate(valloader):
                    volume_valbatch, label_valbatch = sampled_batch['image'], sampled_batch['label']
                    # print(f'volume_valbatch shape: {volume_valbatch.shape}, label_valbatch shape: {label_valbatch.shape}')
                    # print(f'volume_valbatch max and min: {volume_valbatch.max()}, {volume_valbatch.min()}')
                    # print(f'label_valbatch max and min: {label_valbatch.max()}, {label_valbatch.min()}')
                    volume_valbatch, label_valbatch = volume_valbatch.to(device), label_valbatch.to(device)
                    label_valbatch = torch.squeeze(label_valbatch,1)
                    
                    preds_1, _ = model1(volume_valbatch)
                    preds_2, _ = model2(volume_valbatch)
                    
                    preds_1_soft = torch.softmax(preds_1, dim=1)
                    preds_2_soft = torch.softmax(preds_2, dim=1)
                    
                    # calculate all classes
                    dsc1_i = dice_score(preds_1_soft, label_valbatch, num_classes)
                    dsc2_i = dice_score(preds_2_soft, label_valbatch, num_classes)
                    
                    metric_list1 += dsc1_i
                    metric_list2 += dsc2_i
                
                performance1 = metric_list1 / len(db_val)
                performance2 = metric_list2 / len(db_val)
                
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)
                
                performance1 = performance1.item()
                performance2 = performance2.item()

                if performance1 > best_performance1 and iter_num > (max_iterations / 2):
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, 
                                                      round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,'{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)
                    
                if performance2 > best_performance2 and iter_num > (max_iterations / 2):
                    best_performance2 = performance2
                    # save_mode_path = os.path.join(snapshot_path,
                    #                               'model2_iter_{}_dice_{}.pth'.format(
                    #                                   iter_num, 
                    #                                   round(best_performance2, 4)))
                    # save_best = os.path.join(snapshot_path,'{}_best_model2.pth'.format(args.model))
                    # torch.save(model2.state_dict(), save_mode_path)
                    # torch.save(model2.state_dict(), save_best)

                logging.info(
                    'iteration %d : model1_mean_dice : %f ' % (iter_num, performance1))
                logging.info(
                    'iteration %d : model2_mean_dice : %f ' % (iter_num, performance2))
                
                model1.train()
                model2.train()

            '''
            if iter_num % 250 == 0:
                image = volume_batch[0, ...] * 255
                image_save = image.type(torch.int64)
                print('-'*20)
                print(f'image shape: {image.shape}')
                cv2.imwrite(f"{args.root_path}/ssl/assets_unext_{labeled_slice}/image_{iter_num}.png",np.transpose(image_save.data.cpu().numpy(),(1,2,0))) 
                writer.add_image('train/Image', image_save, iter_num)
                
                outputs = torch.argmax(torch.softmax(outputs1, dim=1), dim=1, keepdim=True)
                img1_outputs = np.squeeze(outputs[0]) * 100
                test_output = torch.softmax(outputs1, dim=1)
                img_tensor = test_output[0, ...] * 255
                print('-'*20)
                print(f'outputs1 shape: {outputs1.shape}')
                print(f'outputs shape: {outputs[0].shape}')
                print(f'test_output shape: {test_output.shape}')
                print(f'img_tensor shape: {img_tensor.shape}')
                cv2.imwrite(f"{args.root_path}/ssl/assets_unext_{labeled_slice}/outputs1_{iter_num}.png",img1_outputs.data.cpu().numpy())
                writer.add_image('train/model1_Prediction', outputs[0, ...] * 100, iter_num, dataformats='CHW')
                
                labs = label_batch[0, ...]
                labs_tensor = labs * 100
                print('-'*20)
                print(f'label_batch shape: {label_batch.shape}')
                print(f'labs shape: {labs.shape}')
                cv2.imwrite(f"{args.root_path}/ssl/assets_unet_{labeled_slice}/labs_{iter_num}.png",np.transpose(labs_tensor.data.cpu().numpy(),(1,2,0)))
                writer.add_image('train/GroundTruth', labs_tensor, iter_num, dataformats='CHW')   
            '''

            # if iter_num % 5000 == 0:
            if iter_num == max_iterations:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                # save_mode_path = os.path.join(
                #     snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                # torch.save(model2.state_dict(), save_mode_path)
                # logging.info("save model2 to {}".format(save_mode_path))
                
            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "{}/ssl/model/{}_{}/{}".format(
        args.root_path, args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)    