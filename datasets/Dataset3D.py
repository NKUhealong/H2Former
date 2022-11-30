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

ia.seed(1)    
seq = iaa.Sequential([iaa.Sharpen((0.0, 1.0)),  iaa.ElasticTransformation(alpha=50, sigma=5), iaa.Affine(rotate=(-45, 45)),
                      iaa.Fliplr(0.5), iaa.Flipud(0.5), iaa.Crop(percent=(0, 0.1))], random_order=True)

class MyDataSet(Dataset):
    def __init__(self, root, list_name, train_num, image_size,dataset):
        self.root = root
        self.list_path = self.root + list_name
        self.h, self.w = image_size
        self.dataset = dataset
        self.img_ids = [i_id.strip() for i_id in open(self.list_path)]
        if train_num>700:
            self.img_ids = self.img_ids[:]
        else:
            self.img_ids = self.img_ids[:train_num]
        self.files = []
        for name in self.img_ids:
            if self.dataset == 'skin':
                img_file = os.path.join(self.root, "images/%s.jpg" % name)
            elif self.dataset == 'polyp' or self.dataset == 'DDR':
                img_file = os.path.join(self.root, "images/%s.png" % name)
            else:
                img_file = os.path.join(self.root, "images/%s.jpg" % name)
            label_file = os.path.join(self.root, "masks/%s.png" % name)
            self.files.append({"img": img_file,"label": label_file, "name": name})
        #np.random.shuffle(self.files)
        print("total {} samples".format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        ###  aug 
        
        prob = random.random()
        if prob>0.5:
            segmap = SegmentationMapsOnImage(np.array(label), shape=image.shape)
            image, label = seq(image=image, segmentation_maps=segmap)
            label = label.get_arr()
        
        image = cv2.resize(image,(self.w, self.h),interpolation = cv2.INTER_NEAREST)
        if self.dataset == 'idrid':
            label = cv2.resize(label,(self.w, self.h),interpolation = cv2.INTER_NEAREST)
        else:
            label = cv2.resize(label,(self.w, self.h),interpolation = cv2.INTER_NEAREST)
        name = datafiles["name"]
        #print(np.unique(label))
        image = image[None, :, :]
        image = np.asarray(image, np.float32)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8)).long()
        return image, label
        
  
class RandomGenerator(object):
 
    def __init__(self):
        self.k = 9
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8)).long()
        sample = {'image': image, 'label': label}
        return sample      
        
class testBaseDataSets(Dataset):
    def __init__(self, base_dir, list_name,image_size,dataset,transform):
        self.base_dir = base_dir
        self.sample_list = []
        self.h, self.w = image_size
        self.dataset = dataset
        self.transform = transform
        with open(self.base_dir + list_name, 'r') as f1:
            self.sample_list = f1.readlines()
        self.sample_list = [item.replace('\n', '')  for item in self.sample_list]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx]
        if self.dataset == 'skin':
            image = cv2.imread(self.base_dir + '/images/'+name+'.jpg', cv2.IMREAD_COLOR)
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.dataset == 'polyp' or self.dataset == 'DDR':
            image = cv2.imread(self.base_dir + '/images/'+name+'.png', cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.imread(self.base_dir + '/images/'+name+'.jpg', cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        image = cv2.resize(image,(self.w, self.h),interpolation = cv2.INTER_NEAREST)
        
        
        label = cv2.imread(self.base_dir + '/masks/'+name+'.png', cv2.IMREAD_GRAYSCALE)
        if self.dataset == 'idrid':
            label = cv2.resize(label,(self.w, self.h),interpolation = cv2.INTER_NEAREST)
            #label[label>0] = 1
            #print(np.unique(label))
        else:
            label = cv2.resize(label,(self.w, self.h),interpolation = cv2.INTER_NEAREST)/255
        image = np.asarray(image, np.float32)
        image = image.transpose((2, 0, 1))
        sample = {'image': image, 'label': label}
        sample = self.transform(sample)
        return sample
        
def save_results(pred,save_dir,h,w,j):
    predictions = pred.data.cpu().numpy()
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
