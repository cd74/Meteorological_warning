import sys
import cv2
import shutil
import math
sys.path.insert(0, '.')
#from compiler.ast import flatten
from nowcasting.hko.dataloader import HKOIterator
from nowcasting.config import cfg
import torch
from nowcasting.config import cfg
from nowcasting.models.forecaster import Forecaster
from nowcasting.models.encoder import Encoder
from collections import OrderedDict
from nowcasting.models.model import EF
from torch.optim import lr_scheduler
from nowcasting.models.loss import Weighted_mse_mae
from nowcasting.models.trajGRU import TrajGRU
from nowcasting.train_and_test import train_and_test
import numpy as np
from nowcasting.hko.evaluation import *
from experiments.net_params import *
from nowcasting.models.model import Predictor
from nowcasting.helpers.visualization import save_hko_movie

torch.cuda.set_device(0)
in_seq =5
out_seq =20
encoder = Encoder(encoder_params[0], encoder_params[1]).to(cfg.GLOBAL.DEVICE)
forecaster = Forecaster(forecaster_params[0], forecaster_params[1])
encoder_forecaster = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)

encoder_forecaster.load_state_dict(torch.load('encoder_forecaster_45000.pth'))#save/models/encoder_forecaster_100000.pth'))
torch.save(encoder_forecaster,'full_encoder_forecaster_45000.pth')

criterion = Weighted_mse_mae().to(cfg.GLOBAL.DEVICE)
hko_iter = HKOIterator(pd_path=cfg.HKO_PD.RAINY_TEST,sample_mode="random",seq_len=in_seq + out_seq)

valid_batch, valid_mask, sample_datetimes, _ = \
    hko_iter.sample(batch_size=1)

def filter(img):
    h,w = img.shape
    for i in range(h):
        for j in range(w):
            if ((i-240)**2+(j-240)**2)>=240**2:
                img[i][j]=0
            elif (img[i][j]<=(img.min()+100)):
                img[i][j]=0
#os.environ['CUDA_VISIBLE_DEVICES']="3"
#valid_batch = valid_batch.astype(np.float32) / 255.0
np.save('batch.npy',valid_batch)
np.save('mask.npy',valid_mask)
valid_batch = torch.from_numpy(valid_batch.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
valid_data = valid_batch[:in_seq, ...]
valid_label = valid_batch[in_seq:in_seq + out_seq, ...]
#mask = valid_mask[in_seq:in_seq + out_seq, ...].astype(int)
mask = torch.from_numpy(valid_mask[IN_LEN:IN_LEN + OUT_LEN, ...].astype(int)).to(cfg.GLOBAL.DEVICE)
#torch_valid_data = torch.from_numpy(valid_data).to(cfg.GLOBAL.DEVICE)

# with torch.no_grad():
#     output = encoder_forecaster(valid_data)

# loss = criterion(output, valid_label, mask)
# imlist = []
# output = np.clip(output.cpu().numpy(), 0.0, 1.0)
# out = np.float32(output.tolist())
# out = np.int8(255*out)
##np.save('out.npy',out)
# for j in range(0,10):
#     #img = cv2.applyColorMap(out[j][0][0],cv2.COLORMAP_RAINBOW)
#     img=out[j][0][0]
#     filter(out[j][0][0])
#     cv2.imwrite('testres/'+str(j)+'.png',img)
    
# np.save('out.npy',out)
# base_dir = '.'
# for j in range(0,10):
#     ave = (out[j][0][0]/480/480).sum()
#     avdbz = (ave-0.5)*70/255
#     r = math.e**(avdbz-10*math.log(58.53/10/1.56))
#     if (r>30):
#         print('Time '+str(j)+':rainstorm')
#     elif(r>10):
#         print('Time '+str(j)+':heavy')

# print('loss:',loss)
def calc_r(date,i,img,filename):
   # f=open(filename,'a+')
    lst=np.reshape(img,(1,-1))
    lst.sort()
    ave = lst[0][-5000:].sum()/5000
    avdbz = (ave-0.5)*70/255
    r = 10**((avdbz-10*math.log(58.53,10))/10/1.56)
    if (r>35):
        print(date,'+',i,' ',r,' :rainstorm:4\n')#f.write(date,'+',i,' ',r,' :rainstorm:4\n')
    elif(r>15):
        print(date,'+',i,' ',r,' :heavy:3\n')
    elif(r>7):
        print(date,'+',i,' ',r,' :moderate:2\n')
    elif(r>3):
        print(date,'+',i,' ',r,' :light:1\n')
    else:
        print(date,'+',i,' ',r,' :sunny:0\n')#,'Time '+str(j)+':rainstorm')
  #  f.close()
    
def pred_all():
    hko_iter = HKOIterator(pd_path=cfg.HKO_PD.RAINY_TEST,sample_mode="sequent",seq_len=in_seq + out_seq,stride=25)
    while(hko_iter.use_up==False):
        valid_batch, valid_mask, sample_datetimes, _ = \
            hko_iter.sample(batch_size=1)
        for j in range(0,5):
            calc_r(hko_iter._current_datetime,j,valid_batch[j][0][0],'pred_r.txt')
        valid_batch = torch.from_numpy(valid_batch.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
        valid_data = valid_batch[:in_seq, ...]
        valid_label = valid_batch[in_seq:in_seq + out_seq, ...]
        #mask = valid_mask[in_seq:in_seq + out_seq, ...].astype(int)
        mask = torch.from_numpy(valid_mask[IN_LEN:IN_LEN + OUT_LEN, ...].astype(int)).to(cfg.GLOBAL.DEVICE)
        #torch_valid_data = torch.from_numpy(valid_data).to(cfg.GLOBAL.DEVICE)
        with torch.no_grad():
            output = encoder_forecaster(valid_data)
        #valid_data = np.int16(output.cpu().numpy().tolist())       
        loss = criterion(output, valid_label, mask)
        imlist = []
        output = np.clip(output.cpu().numpy(), 0.0, 1.0)
        out = np.float32(output.tolist())
        out = np.int16(255*out)
        r=0
        for j in range(0,20):
            #max_r=max(max_r,out[j][0][0].max())
    #img = cv2.applyColorMap(out[j][0][0],cv2.COLORMAP_RAINBOW)
            # img=out[j][0][0]
            # filter(out[j][0][0])
            #ave = (out[j][0][0]/480/480).sum()   
            calc_r(hko_iter._current_datetime,j+5,out[j][0][0],'pred_r.txt')        
        #     lst=np.reshape(out[j][0][0],(1,-1))
        #     lst.sort()
        #     ave = lst[0][-5000:].sum()/5000
        #     avdbz = (ave-0.5)*70/255
        #     r = max(r,10**((avdbz-10*math.log(58.53,10))/10/1.56))
        # if (r>35):
        #     print(hko_iter._current_datetime,'+2.5h,max r=',r,' :rainstorm:4\n')
        # elif(r>15):
        #     print(hko_iter._current_datetime,'+2.5h,max r=',r,' :heavy:3\n')
        # elif(r>7):
        #     print(hko_iter._current_datetime,'+2.5h,max r=',r,' :moderate:2\n')
        # elif(r>3):
        #     print(hko_iter._current_datetime,'+2.5h,max r=',r,' :light:1\n')
        # else:
        #     print(hko_iter._current_datetime,'+2.5h,max r=',r,' :sunny:0\n')#,'Time '+str(j)+':rainstorm')
            #elif(r>10):
            #    print(hko_iter.end_time,'Time '+str(j)+':heavy')

def pickdate():
    dates = set()
    f = open('datepred.txt','r')
    for line in f:
        date = line.split(' ')
        dates.add(date[0])
    print(sorted(dates))

def real_all():
    hko_iter = HKOIterator(pd_path=cfg.HKO_PD.RAINY_TEST,sample_mode="sequent",seq_len=in_seq + out_seq,stride=25)
    while(hko_iter.use_up==False):
        valid_batch, valid_mask, sample_datetimes, _ = \
                hko_iter.sample(batch_size=1)
        for j in range(0,25):
            calc_r(hko_iter._current_datetime,j,valid_batch[j][0][0],'pred_r.txt')
#pickdate()
pred_all()   
#real_all() 

# S*B*1*H*W
# label = valid_label[:, 0, 0, :, :]
# output = output[:, 0, 0, :, :]
# mask = mask[:, 0, 0, :, :].astype(np.uint8)
# save_hko_movie(label, sample_datetimes[0], mask, masked=True,
#                        save_path=os.path.join(base_dir, 'ground_truth.mp4'))
# save_hko_movie(output, sample_datetimes[0], mask, masked=True,
#                save_path=os.path.join(base_dir, 'pred.mp4'))
