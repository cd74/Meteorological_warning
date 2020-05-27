from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import cv2
import os
import os.path
import errno
import numpy as np
import torch
import codecs

#demo train
# for j, (seq,tag) in enumerate(train_loader):
#             seq=seq[[0,1,2,3,4,5,6,7,8]]
#             tag=tag[[4,9,14,19]]
#             seq=seq/255.0
#             tag=tag/255.0
#             input = torch.from_numpy(seq.astype(np.float32)).to(cfg.GLOBAL.DEVICE) 

class HKO_dataloader(data.Dataset):

    def __init__(self, train_path, test_path, in_len,out_len, train=True, 
                    transform=None, target_transform=None):
        self.train_path = train_path
        self.test_path = test_path
        self.in_len = in_len
        self.out_len = out_len
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if self.train:
            self.train_data = np.load(self.train_path)
               # os.path.join(self.train_path, self.processed_folder, self.training_file))
        else:
            self.test_data = np.load(self.test_path)
               #os.path.join(self.test_path, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (seq, target) where sampled sequences are splitted into a seq
                    and target part
        """
        if self.train:
            seq, target = self.train_data[:self.in_len,index], self.train_data[self.in_len:self.in_len+self.out_len,index]
        else:
            seq, target = self.test_data[:self.in_len,index], self.test_data[self.in_len:self.in_len+self.out_len,index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # seq = Image.fromarray(seq.numpy(), mode='L')
        # target = Image.fromarray(target.numpy(), mode='L')

        # if self.transform is not None:
        #     seq = self.transform(seq)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return seq, target

    def __len__(self):
        if self.train:
            return len(self.train_data[0])
        else:
            return len(self.test_data[0])

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Train/test: {}\n'.format(tmp)
        fmt_str += '    Trainset path: {}\n'.format(self.train_path)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str




