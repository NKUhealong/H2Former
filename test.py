from dataset import *
from torchvision import transforms
import copy
import time
import datetime
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis

from models.H2former import *
from dataset import *
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
num_classes = 2
batch_size = 1
image_size = (512,512)
save_dir='./result/'

base_dir = './data/skin/test/'   # polyp, idrid, skin
dataset = 'skin'

db_val = testBaseDataSets(base_dir, 'test.txt',image_size,dataset,transform=transforms.Compose([RandomGenerator()]))
valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False,num_workers=0)
    
model_name='res34_swin_MS_skin'    
model = res34_swin_MS(image_size[0],2)  

for k in range(75,76,3): 
    print('./new/'+model_name+str(k)+'.pth')
    model.load_state_dict(torch.load('./new/'+model_name+str(k)+'.pth'))
    model.cuda()
    model.eval()
    j = 0
    evaluator = Evaluator()
    start_time = time.time()
    with torch.no_grad():
        for sampled_batch in valloader:
            images, labels = sampled_batch['image'], sampled_batch['label']
            images, labels = images.cuda(),labels.cuda()

            predictions = model(images)
            pred = predictions[0,1,:,:]
            evaluator.update(pred, labels[0,:,:].float())

            for i in range(batch_size):
                labels = labels.cpu().numpy()
                images = images[i].cpu().numpy()
                label = (labels[i]*255)
                pred = pred.cpu().numpy()
                #total_img = np.concatenate((label,pred[:,:]*255),axis=1)
                #cv2.imwrite(save_dir+'Pre'+str(j)+'.jpg',pred[:,:]*255)
                #cv2.imwrite(save_dir+'GT'+str(j)+'.jpg',label)
                #cv2.imwrite(save_dir+'image'+str(j)+'.jpg',images.transpose(1, 2, 0)[:,:,::-1])
                j=j+1

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Test time {}'.format(total_time_str))  
    evaluator.show()
    
