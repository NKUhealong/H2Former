import  torch
import numpy as np
import os
import time
import datetime
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import cv2
import math
from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis

from models.H2former import *
from dataset import *
from utils import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 2022 
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

mode = 'train'
batch_size =  2
base_lr = 0.0001
num_classes = 5
image_size = (960, 960)
base_dir = './data/idrid/'   # polyp, idrid, skin 900
dataset = 'idrid' 
train_num = 383  
max_epoch = 92

model = res34_swin_MS(image_size[0],num_classes) 
model_name = 'H2former'
test_model_name = 'H2former'

print('model parameters: ', sum(p.numel() for p in model.parameters())/1e6,'M' )

model_dict = model.state_dict()
pre_dict = torch.load('./resnet34.pth') 
matched_dict = {k: v for k, v in pre_dict.items() if k in model_dict and v.shape==model_dict[k].shape}
print('matched keys:', len(matched_dict))
model_dict.update(matched_dict)
model.load_state_dict(model_dict)

model.cuda()

ce_loss_func = nn.CrossEntropyLoss()
dice_loss_func = DiceLoss(num_classes)

optimizer = optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=3e-5, amsgrad=False)
#optimizer = optim.Adam(model.parameters(), betas=(0.9,0.99), lr=base_lr, weight_decay=0.0001)
#optimizer = optim.SGD(model.parameters(), lr=base_lr,momentum=0.9, weight_decay=0.0001)

train_data = MyDataSet(base_dir+'train/','train.txt',train_num, image_size,dataset)
trainloader = DataLoader(train_data, batch_size=batch_size,shuffle=True, num_workers=8,pin_memory = True)

print('train len:', len(trainloader))

iter_num = 0
max_iterations =  max_epoch * len(trainloader) 
if mode == 'test':
    max_epoch = 1
for epoch_num in range(max_epoch):
    model.train()  
    train_acc = 0
    train_loss = 0
    test_acc =0
    start_time = time.time()
    if mode == 'train':
        for batch_images, batch_labels in trainloader:

            batch_images, batch_labels = batch_images.cuda(),batch_labels.cuda()
            out = model(batch_images)
            outputs_soft = torch.softmax(out, dim=1)
            loss = ce_loss_func(out, batch_labels.long()) + dice_loss_func(out, batch_labels.long())
            #loss_margin = torch.mean(1-(outputs_soft[:,1,:,:] - outputs_soft[:,0,:,:])*batch_labels)
            loss = loss# + loss_margin
            train_loss = loss.data.cpu().numpy() + train_loss

            prediction = torch.max(out,1)[1]
            train_correct = (prediction == batch_labels).float().cpu().numpy().mean()
            train_acc = train_acc + train_correct

            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step()  
            lr = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            iter_num = iter_num + 1
    else:
        print('test model name:', test_model_name)
        model.load_state_dict(torch.load('./new/'+test_model_name+'.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    model.eval()
    if epoch_num%3 == 0: 
        print('Epoch: {} / {}'.format(epoch_num, max_epoch),'  train time {}'.format(total_time_str),
              ' train Loss {:4f}'.format(train_loss/len(trainloader)),' train Acc {:4f}'.format(train_acc/len(trainloader)),'  LR {:4f}'.format(lr))
        if epoch_num>30:
            torch.save(model.state_dict(), './new/'+model_name+str(epoch_num)+'.pth')  
        
        test_batch_size = 1
        save_dir='./result/'
        db_val = testBaseDataSets(base_dir+'valid/', 'valid.txt',image_size,dataset,transform=transforms.Compose([RandomGenerator()]))
        valloader = DataLoader(db_val, batch_size=test_batch_size, shuffle=False,num_workers=0)
        j = 0
        if dataset == 'idrid':
            evaluator_EX= Evaluator()
            evaluator_HE= Evaluator()
            evaluator_MA= Evaluator()
            evaluator_SE= Evaluator()
            with torch.no_grad():
                for sampled_batch in valloader:
                    images, labels = sampled_batch['image'], sampled_batch['label']
                    images, labels = images.cuda(),labels.cuda()

                    pred = model(images)
                    probs = pred.cpu().numpy()
                    predictions = torch.argmax(pred, dim=1)
                    save_results(predictions,save_dir,image_size[0],image_size[1],j)
                    j = j+1
                    
                    pred = F.one_hot(predictions.long(), num_classes=num_classes)
                    new_labels =  F.one_hot(labels.long(), num_classes=num_classes)

                    evaluator_EX.update(pred[0,:,:,1], new_labels[0,:,:,1].float())
                    evaluator_HE.update(pred[0,:,:,2], new_labels[0,:,:,2].float())
                    evaluator_MA.update(pred[0,:,:,3], new_labels[0,:,:,3].float())
                    evaluator_SE.update(pred[0,:,:,4], new_labels[0,:,:,4].float())
            MAE_ex, Recall_ex, Pre_ex, Acc_ex, Dice_ex, IoU_ex = evaluator_EX.show(False)
            MAE_he, Recall_he, Pre_he, Acc_he, Dice_he, IoU_he = evaluator_HE.show(False)
            MAE_ma, Recall_ma, Pre_ma, Acc_ma, Dice_ma, IoU_ma = evaluator_MA.show(False)
            MAE_se, Recall_se, Pre_se, Acc_se, Dice_se, IoU_se = evaluator_SE.show(False)
            MAE =  (MAE_ex + MAE_he + MAE_ma +MAE_se )/4
            Recall = (Recall_ex + Recall_he + Recall_ma +Recall_se)/4
            Pre =  (Pre_ex + Pre_he + Pre_ma +Pre_se )/4
            Acc =  (Acc_ex + Acc_he + Acc_ma +Acc_se )/4
            Dice =  (Dice_ex + Dice_he + Dice_ma +Dice_se)/4 
            IoU =  (IoU_ex + IoU_he + IoU_ma +IoU_se)/4 
            print("MAE:", "%.2f" % MAE,"  Recall: ", "%.2f" % Recall, " Pre:", "%.2f" % Pre," Acc:", "%.2f" % Acc," Dice:", "%.2f" % Dice," IoU:" , "%.2f" % IoU) 
            print('\n')        
            
        else:
            evaluator = Evaluator()        