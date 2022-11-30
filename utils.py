import cv2
import torch
import os
import random
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
        
class Evaluator:
    def __init__(self, cuda=True):
        self.cuda = cuda
        self.MAE = list()
        self.Recall = list()
        self.Precision = list()
        self.Accuracy = list()
        self.Dice = list()       
        self.IoU_polyp = list()

    def evaluate(self, pred, gt):
        
        pred_binary = (pred >= 0.5).float().cuda()
        pred_binary_inverse = (pred_binary == 0).float().cuda()

        gt_binary = (gt >= 0.5).float().cuda()
        gt_binary_inverse = (gt_binary == 0).float().cuda()
        
        MAE = torch.abs(pred_binary - gt_binary).mean().cuda(0)
        TP = pred_binary.mul(gt_binary).sum().cuda(0)
        FP = pred_binary.mul(gt_binary_inverse).sum().cuda(0)
        TN = pred_binary_inverse.mul(gt_binary_inverse).sum().cuda(0)
        FN = pred_binary_inverse.mul(gt_binary).sum().cuda(0)

        if TP.item() == 0:
            TP = torch.Tensor([1]).cuda(0)
        # recall
        Recall = TP / (TP + FN)
        # Precision or positive predictive value
        Precision = TP / (TP + FP)
        #Specificity = TN / (TN + FP)
        # F1 score = Dice
        Dice = 2 * Precision * Recall / (Precision + Recall)
        # Overall accuracy
        Accuracy = (TP + TN) / (TP + FP + FN + TN)
        # IoU for poly
        IoU_polyp = TP / (TP + FP + FN)

        return MAE.data.cpu().numpy().squeeze(), Recall.data.cpu().numpy().squeeze(), Precision.data.cpu().numpy().squeeze(), Accuracy.data.cpu().numpy().squeeze(), Dice.data.cpu().numpy().squeeze(), IoU_polyp.data.cpu().numpy().squeeze()

        
    def update(self, pred, gt):
        mae, recall, precision, accuracy, dice, ioU_polyp = self.evaluate(pred, gt)        
        self.MAE.append(mae)
        self.Recall.append(recall)
        self.Precision.append(precision)
        self.Accuracy.append(accuracy)
        self.Dice.append(dice)       
        self.IoU_polyp.append(ioU_polyp)

    def show(self,flag = True):
        if flag == True:
            
            print("MAE:", "%.2f" % (np.mean(self.MAE)*100),"  Recall:", "%.2f" % (np.mean(self.Recall)*100), "  Pre:", "%.2f" % (np.mean(self.Precision)*100),
                  "  Acc:", "%.2f" % (np.mean(self.Accuracy)*100),"  Dice:", "%.2f" % (np.mean(self.Dice)*100),"  IoU:" , "%.2f" % (np.mean(self.IoU_polyp)*100))   
            print('\n')
        
        return np.mean(self.MAE)*100,np.mean(self.Recall)*100,np.mean(self.Precision)*100,np.mean(self.Accuracy)*100,np.mean(self.Dice)*100,np.mean(self.IoU_polyp)*100

def save_results(pred,save_dir,h,w,j):
    predictions = pred.cpu().numpy()
    test_num= len(predictions)
    for i in range(test_num):
        pred = predictions[i]        
        pred_vis = np.zeros((h,w,3),np.uint8)
        pred_vis[pred==1]=[255,0,0]
        pred_vis[pred==2]=[0,255,0]
        pred_vis[pred==3]=[0,0,255]
        pred_vis[pred==4]=[255,0,255]
        
        cv2.imwrite(save_dir+'Pred'+str(j)+'.png',pred_vis[:,:,::-1])
        
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        #print(input_tensor.size())
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            temp_prob = torch.unsqueeze(temp_prob, 1)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        #print(inputs.size(),target.size())
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
