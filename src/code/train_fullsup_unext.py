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
from config import get_config
from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler)
# from dataloaders.dataset_fetalhead import BaseDataSets_HC18 as USDataSets
# from dataloaders.dataset_fetalhead import BaseDataSets_ESTT as USDataSets
# from dataloaders.dataset_fetalhead import BaseDataSets_PSFHS as USDataSets
# from dataloaders.dataset_fetalhead import BaseDataSets_FetalAbdomen as USDataSets
from dataloaders.dataset_fetalhead import BaseDataSets_BUSI as USDataSets
from networks.archs import UNext as Unet2D
from utils import losses, ramps #, metrics
from utils.dice import DSC
from utils.metrics import dice_score
from utils.losses import ConstraLoss, info_nce_loss, softmax_kl_loss


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
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()
config = get_config(args)
# print(config)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


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

    model1 = Unet2D(in_chns=3, 
                    out_channels=args.num_classes,
                    img_size=args.patch_size[0],
                    num_classes=args.num_classes).to(device)
    
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
        transforms.Resize((args.patch_size[0], args.patch_size[1]), transforms.InterpolationMode.NEAREST),
    ])
    
    db_train_labled = USDataSets(base_dir=args.root_path, split="suptrain", num=labeled_slice, islabeled=True, 
                                         transform=tr_transforms, default_transform=tensor_transforms) 
    
    db_train = db_train_labled
    total_slices = len(db_train)
    logging.info(f'total lenght of train data: {len(db_train)}')
    
    db_val = USDataSets(base_dir=args.root_path, split="val", num=val_size, 
                        transform=None, default_transform=tensor_transforms)
    # samples = random.sample(range(1, len(db_val)), val_size)
    # db_val = Subset(db_val, samples)
    logging.info(f'total lenght of val data: {len(db_val)}')

    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    
    # print("labeled_idxs: {}, unlabeled_idxs: {}".format(labeled_idxs, unlabeled_idxs))

    trainloader = DataLoader(db_train, batch_size=batch_size, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    
    model1.train()
    
    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    time1 = int(time.time())
    writer = SummaryWriter(snapshot_path + f'/log_{time1}')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    evalue = DSC()  # metric to find best model
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
            
            # print(f'volume_batch shape: {volume_batch.shape}')
            # print(f'label_batch shape: {label_batch.shape}')
            # print(f'volume_batch max: {volume_batch.max()}')
            # print(f'volume_batch min: {volume_batch.min()}')
            # print(f'label_batch max: {label_batch.max()}')
            # print(f'label_batch min: {label_batch.min()}')

            outputs1, _ = model1(volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            # print(f'outputs1 shape: {outputs1.shape}') # B, 3, 224, 224
            # print(f'outputs2 shape: {outputs1.shape}') # B, 3, 224, 224
            # print(f'label_batch shape: {label_batch.shape}') # B, 1, 224, 224
            
            loss1 = ce_loss(outputs1[:args.labeled_bs], torch.squeeze(label_batch[:args.labeled_bs],1).long()) + \
                dice_loss(outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs])
            
            loss = loss1

            optimizer1.zero_grad()

            loss.backward()

            optimizer1.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/model1_loss', loss1, iter_num)
            logging.info('iteration %d : model1 loss : %f' % (iter_num, loss.item()))
            
            if iter_num > 0 and iter_num % (labeled_slice / args.labeled_bs) == 0:
                model1.eval()
                
                metric_list1 = 0.0
                
                for i_batch, sampled_batch in enumerate(valloader):
                    volume_valbatch, label_valbatch = sampled_batch['image'], sampled_batch['label']
                    volume_valbatch, label_valbatch = volume_valbatch.to(device), label_valbatch.to(device)
                    label_valbatch = torch.squeeze(label_valbatch,1)

                    preds_1, _ = model1(volume_valbatch)
                    # print(f'line 250: preds_1 max: {preds_1.max()}, preds_1 min: {preds_1.min()}')
                    
                    preds_1_soft = torch.softmax(preds_1, dim=1)
                    # print(f'line 253: preds_1 max: {preds_1_soft.max()}, preds_1 min: {preds_1_soft.min()}')
                    
                    # print(preds_1_soft.shape, label_valbatch.shape)
                    dsc1_i = dice_score(preds_1_soft, label_valbatch, num_classes)
                    # print(f'dsc1_i: {dsc1_i}')

                    metric_list1 += dsc1_i

                performance1 = metric_list1 / len(db_val)
                
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)
            
                performance1 = performance1.item()

                if performance1 > best_performance1 and iter_num > (max_iterations / 2):
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,'model1_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,'{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                logging.info(
                    'iteration %d : model1_mean_dice : %f ' % (iter_num, performance1))
                
                model1.train()
            
            '''
            if iter_num % 1000 == 0:
                image = volume_batch[0, ...] * 255
                image_save = image.type(torch.int64)
                print('-'*20)
                print(f'image shape: {image.shape}')
                cv2.imwrite(f"{args.root_path}/ssl/assets_unext_fullsup/image_{iter_num}.png",np.transpose(image_save.data.cpu().numpy(),(1,2,0))) 
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
                cv2.imwrite(f"{args.root_path}/ssl/assets_unext_fullsup/outputs1_{iter_num}.png",img1_outputs.data.cpu().numpy())
                writer.add_image('train/model1_Prediction', outputs[0, ...] * 100, iter_num, dataformats='CHW')
                
                labs = label_batch[0, ...]
                labs_tensor = labs * 100
                print('-'*20)
                print(f'label_batch shape: {label_batch.shape}')
                print(f'labs shape: {labs.shape}')
                cv2.imwrite(f"{args.root_path}/ssl/assets_unext_fullsup/labs_{iter_num}.png",np.transpose(labs_tensor.data.cpu().numpy(),(1,2,0)))
                writer.add_image('train/GroundTruth', labs_tensor, iter_num, dataformats='CHW')   
            '''

            # if iter_num % 5000 == 0:
            if iter_num == max_iterations:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))
                
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