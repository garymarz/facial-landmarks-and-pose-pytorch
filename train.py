#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import logging
from pathlib import Path
import time
import os
from tqdm import tqdm
import numpy as np
import torch

from torch.utils import data
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from dataset.datasets import WLFWDatasets, Pose_300W_LP
from models.pfld import PFLDInference, PoseNet2
from pfld.loss import PFLDLoss, GeodesicLoss
from pfld.utils import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def train(train_loader, pfld_backbone, auxiliarynet,   criterion, optimizer,
          epoch, train_landmarks):
    losses = AverageMeter()

    weighted_loss, loss = None, None
    if train_landmarks:
        for img, landmark_gt, attribute_gt, euler_angle_gt in tqdm(train_loader):
            img = img.to(device)
            attribute_gt = attribute_gt.to(device)
            landmark_gt = landmark_gt.to(device)
            euler_angle_gt = euler_angle_gt.to(device)
            pfld_backbone = pfld_backbone.to(device)
            auxiliarynet = auxiliarynet.to(device)
            features, landmarks = pfld_backbone(img)
            angle = auxiliarynet(features)
            weighted_loss = 0

            weighted_loss, loss, loss_euler_angle = criterion(attribute_gt, landmark_gt,
                                                euler_angle_gt, angle, landmarks,
                                                args.train_batchsize)
            
            weighted_loss += 0.8*loss
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            losses.update(loss.item())
        print('weighted_loss',weighted_loss)   
        return weighted_loss, loss , loss_euler_angle

    else:
        for img, _ , _ , euler_angle_gt,labels,_ in train_loader:
            img = img.to(device)
            #attribute_gt = attribute_gt.to(device)
            #landmark_gt = landmark_gt.to(device)
            euler_angle_gt = euler_angle_gt.to(device)
            labels = labels.to(device)
            pfld_backbone = pfld_backbone.to(device)
            auxiliarynet = auxiliarynet.to(device)
            features, _ = pfld_backbone(img)
            angle = auxiliarynet(features)
            loss_euler_angle = 0
            #print(labels.shape,angle.shape)
            mae = torch.nn.L1Loss()

            # compute the loss (mean absolute error)
            loss_euler_angle = mae(labels,angle)
            #loss_euler_angle = smoothL1(labels,angle)
            #loss_euler_angle = criterion(euler_angle_gt, angle)
            optimizer.zero_grad()
            loss_euler_angle.backward()
            optimizer.step()
        
        #print('loss',weighted_loss, 'loss_euler',loss_euler_angle)
        #weighted_loss+=loss_euler_angle*0.8
        print('loss_euler_angle',loss_euler_angle)
        return _ , _ , loss_euler_angle


def validate(wlfw_val_dataloader, pfld_backbone, auxiliarynet, criterion):
    pfld_backbone.eval()
    auxiliarynet.eval()
    losses = []
    with torch.no_grad():
        for img, landmark_gt, attribute_gt, euler_angle_gt in wlfw_val_dataloader:
            img = img.to(device)
            attribute_gt = attribute_gt.to(device)
            landmark_gt = landmark_gt.to(device)
            euler_angle_gt = euler_angle_gt.to(device)
            pfld_backbone = pfld_backbone.to(device)
            auxiliarynet = auxiliarynet.to(device)
            _, landmark = pfld_backbone(img)
            loss = torch.mean(torch.sum((landmark_gt - landmark)**2, axis=1))
            losses.append(loss.cpu().numpy())
    print("===> Evaluate:")
    print('Eval set: Average loss: {:.4f} '.format(np.mean(losses)))
    return np.mean(losses)


def main(args):
    # Step 1: parse args config
    logging.basicConfig(
        format=
        '[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode='w'),
            logging.StreamHandler()])
    
    print_args(args)

    # Step 2: model, criterion, optimizer, scheduler
    pfld_backbone = PFLDInference().to(device)
    #auxiliarynet = AuxiliaryNet().to(device)
    auxiliarynet = PoseNet2().to(device)
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    transformations = transforms.Compose([transforms.Resize(120),
                                          transforms.RandomCrop(112),
                                          transforms.ToTensor(),
                                          normalize])
    if args.train_landmarks:
        print('Train landmarks')
        criterion = PFLDLoss()
        optimizer = torch.optim.Adam([{
            'params': pfld_backbone.parameters()
             },{
            'params': auxiliarynet.parameters()
             }],lr=args.base_lr,weight_decay=args.weight_decay)
        wdataset = WLFWDatasets(args.dataroot, transformations)
        milestones = [10, 20]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.5)
        
    else:
        print('Train euler_angle')
        pfld_backbone.eval()
        criterion = GeodesicLoss()
        optimizer = torch.optim.Adam([{
            'params': pfld_backbone.parameters()
             },{
            'params': auxiliarynet.parameters()
             }],lr=args.base_lr,weight_decay=args.weight_decay)
        wdataset = Pose_300W_LP('data/300W_LP', 'data/300W_LP/files.txt',transformations)
        #milestones = np.arange(num_epochs)
        milestones = [10, 20]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.5)

    
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', patience=args.lr_patience, verbose=True)
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        pfld_backbone.load_state_dict(checkpoint["pfld_backbone"])
        #auxiliarynet.load_state_dict(checkpoint["auxiliarynet"])
        args.start_epoch = checkpoint["epoch"]

    # step 3: data
    # argumetion
    dataloader = DataLoader(wdataset,
                            batch_size=args.train_batchsize,
                            shuffle=True,
                            num_workers=args.workers,
                            drop_last=False)
    
#     normalize = transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225])

#     transformation = transforms.Compose([transforms.Resize(112),
#                                           transforms.ToTensor(),
#                                           normalize])
#     wlfw_val_dataset = WLFWDatasets(args.val_dataroot, transformation)
#     wlfw_val_dataloader = DataLoader(wlfw_val_dataset,
#                                      batch_size=args.val_batchsize,
#                                      shuffle=False,
#                                      num_workers=args.workers)

    # step 4: run
#    writer = SummaryWriter(args.tensorboard)
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        weighted_train_loss, train_loss, loss_euler_angle = train(dataloader, pfld_backbone,
                                                auxiliarynet, criterion,
                                                optimizer, epoch, args.train_landmarks)
        scheduler.step()
        
        filename = os.path.join(str(args.snapshot),
                                "checkpoint_epoch_" + str(epoch) + '.pth.tar')
        save_checkpoint(
            {
                'epoch': epoch,
                'pfld_backbone': pfld_backbone.state_dict(),
                'auxiliarynet': auxiliarynet.state_dict()
            }, filename)

#        val_loss = validate(wlfw_val_dataloader, pfld_backbone, auxiliarynet,
#                             criterion)

#         scheduler.step(val_loss)
#         writer.add_scalar('data/weighted_loss', weighted_train_loss,loss_euler_angle, epoch)
#         writer.add_scalars('data/loss', {
#             'val loss': val_loss,
#             'train loss': train_loss
#         }, epoch)
#     writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='pfld')
    # general
    parser.add_argument('-j', '--workers', default=0, type=int)
    parser.add_argument('--devices_id', default='0', type=str)  #TBD
    parser.add_argument('--test_initial', default='false', type=str2bool)  #TBD
    parser.add_argument('--train_landmarks', default= True, type=str2bool)
    # training
    ##  -- optimizer
    parser.add_argument('--base_lr', default=0.00001, type=int)
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float)

    # -- lr
    parser.add_argument("--lr_patience", default=40, type=int)

    # -- epoch
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=600, type=int)

    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument('--snapshot',
                        default='./checkpoint/snapshot/',
                        type=str,
                        metavar='PATH')
    parser.add_argument('--log_file',
                        default="./checkpoint/train.logs",
                        type=str)
    parser.add_argument('--tensorboard',
                        default="./checkpoint/tensorboard",
                        type=str)
    parser.add_argument(
        '--resume',
        default='',
        type=str,
        metavar='PATH')

    # --dataset
    parser.add_argument('--dataroot',
                        default='./data/train_data/list.txt',
                        type=str,
                        metavar='PATH')
    parser.add_argument('--val_dataroot',
                        default='./data/test_data/list.txt',
                        type=str,
                        metavar='PATH')
    parser.add_argument('--train_batchsize', default=256, type=int)
    parser.add_argument('--val_batchsize', default=256, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
