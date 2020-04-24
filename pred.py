import sys
import cv2
import shutil
import math
sys.path.insert(0, '.')
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

in_seq =10
out_seq =10
encoder = Encoder(encoder_params[0], encoder_params[1]).to(cfg.GLOBAL.DEVICE)
forecaster = Forecaster(forecaster_params[0], forecaster_params[1])
encoder_forecaster = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)

encoder_forecaster.load_state_dict(torch.load('encoder_forecaster_45000.pth'))#save/models/encoder_forecaster_100000.pth'))
torch.save(encoder_forecaster,'full_encoder_forecaster_45000.pth')

hko_iter = HKOIterator(pd_path=cfg.HKO_PD.RAINY_TEST,sample_mode="random",seq_len=in_seq + out_seq)

valid_batch, valid_mask, sample_datetimes, _ = \
    hko_iter.sample(batch_size=1)

os.environ['CUDA_VISIBLE_DEVICES']="3"
valid_batch = valid_batch.astype(np.float32) / 255.0
valid_data = valid_batch[:in_seq, ...]
valid_label = valid_batch[in_seq:in_seq + out_seq, ...]
mask = valid_mask[in_seq:in_seq + out_seq, ...].astype(int)
torch_valid_data = torch.from_numpy(valid_data).to(cfg.GLOBAL.DEVICE)

with torch.no_grad():
    output = encoder_forecaster(torch_valid_data)

imlist = []
output = np.clip(output.cpu().numpy(), 0.0, 1.0)
out = np.float32(output.tolist())
out = np.int8(255*out)
for j in range(0,10):
    #img = cv2.applyColorMap(out[j][0][0],cv2.COLORMAP_RAINBOW)
    img=out[j][0][0]
    cv2.imwrite('testres/'+str(j)+'.png',img)
base_dir = '.'
for j in range(0,10):
    ave = (out[j][0][0]/480/480).sum()
    avdbz = (ave-0.5)*70/255
    r = math.e**(avdbz-10*math.log(58.53/10/1.56))
    if (r>30):
        print('Time '+str(j)+':rainstorm')
    elif(r>10):
        print('Time '+str(j)+':heavy')
# S*B*1*H*W
# label = valid_label[:, 0, 0, :, :]
# output = output[:, 0, 0, :, :]
# mask = mask[:, 0, 0, :, :].astype(np.uint8)
# save_hko_movie(label, sample_datetimes[0], mask, masked=True,
#                        save_path=os.path.join(base_dir, 'ground_truth.mp4'))
# save_hko_movie(output, sample_datetimes[0], mask, masked=True,
#                save_path=os.path.join(base_dir, 'pred.mp4'))
