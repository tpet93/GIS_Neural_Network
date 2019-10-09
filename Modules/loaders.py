'''
Copyright (c) 2019 Tony Peter https://github.com/tpet93

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import os
import glob
import torch
import random
import numpy as np
import tifffile as tiff

import matplotlib.gridspec as gridspec
from sklearn.metrics import jaccard_score as jsc
import matplotlib.pyplot as plt



from .utils import *


import h5py

from torch.utils.data import Dataset

class HDF5Dataset(Dataset):

  def __init__(self, datasetpath, transform=None):
    #creates a Pytorch Dataset class based on a hdf5 file generated previously.
    #this improves loading time during training.
    #transform can be used to apply noise or other transformations to the input image (not the class) on each load
    

    self.f = h5py.File(datasetpath+'dataset.hdf5', "r")
    self.images = self.f['images']
    self.labels = self.f['labels']

    self.datasetpath = datasetpath
    self.transform = transform

  def __len__(self):
    return self.images.shape[0]

  def __getitem__(self, idx):

    _img = self.images[idx]
    _lbl = self.labels[idx]

    img_1,lbl_1 = Randomtransform(_img,_lbl)

    if self.transform is not None:
      img = self.transform(img_1)
    else:
      img = torch.from_numpy(img_1.copy())#applying a torch transforms seems to flip or rotated the image, this workaround ensures the same orriention regardles of whether a trasform was used.
      img = img.transpose(0, 2)  #output (batch,class,x,y)
      img = img.transpose(1, 2)  #output (batch,class,x,y)


    lbl = torch.from_numpy(lbl_1.copy())



    return img,lbl


class tifimagedataset(Dataset):
  #creates a pytorch dataset based on a folder of tiff images.
  def __init__(self, datasetpath, transform=None):
    

    self.filenames = [os.path.basename(x) for x in glob.glob(datasetpath+"*.tif")]
    self.datasetpath = datasetpath
    self.transform = transform
  def __len__(self):
    return len(self.filenames)
    # override the __getitem__ method. this is the method dataloader calls
  def __getitem__(self, index):
      # this is what ImageFolder normally returns 

      img_ = tiff.imread(self.datasetpath + self.filenames[index])
      
      if self.transform is not None:
          img_ = self.transform(img_)
      img = np.array(img_.copy())
      singleband = len(img.shape) == 2
      if(singleband):
          img = np.expand_dims(img, axis=2)

      return img , self.filenames[index]




def show_image(model, device,weightsmat,dataloader,ps,bands = (0),dividers = [1]):
  #shows an image from a training or eveluation dataloader. 
  #usefull for previewing input band modificataions/ model performance.
  #TODO there seems to be an excessive amount of transposes, due to the classes needing to be at index 0  or 2 and flipping that occurs between numpy and torch tensors.


  ps = int(ps)
  inputs, classes = next(iter(dataloader))  
  tmg = inputs[0].unsqueeze(dim = 0).to(device)

  npinputs =np.array(inputs[0].transpose(0,2).transpose(0,1))#create a numpy copy of the input images
  npclasses =np.array(classes[0].squeeze(dim=0))#create a numpy copy of the input images

  model.eval()
  out1 = model(tmg.float()).squeeze(dim=0)
  model.train()
  
  out2 = out1.cpu().detach().numpy()
  temp = np.swapaxes(out2,2,0)
  dotprod = np.dot( temp, np.asarray(weightsmat))

  if len(dotprod.shape) > 2:# calc the dotproduct of the output values and our weights matrix.
    newout = np.swapaxes(dotprod,0,2)
  else:
    newout = out2

  b_ =newout.argmax(0)# get the class with the highest likelyhood.


  true_seg = decode_segmap(npclasses)#generate coloured maps
  pred_seg = decode_segmap(b_)#generate coloured maps
  
  lbl = classes[0].reshape(-1)
  target = b_.reshape(-1)
  
  zeroidxs = np.where(lbl==0)#ignore areas with a label value of zero in our accuracy calcs
  lbl=np.delete(lbl, zeroidxs)
  target=np.delete(target, zeroidxs)

  npinputs = npinputs[ps:-ps,ps:-ps,bands]#select the most useful bands to display
  
  nc = newout.shape[0]

  if(nc >1):
    jaccard = jsc(target,lbl,average='micro')#TODO better averaging
    print('Jaccard: ',jaccard)

  figure = plt.figure(figsize=(8, 8))
  plt.subplot(1, 3, 1)
  plt.title('Input Image')
  plt.axis('off')

  npinputs = npinputs/dividers
  try:
    npinputs = np.squeeze(npinputs, axis=2)#if singleband then squeeze
  except ValueError:
    pass #do nothing only want to speeze on 1 len dim


  if(nc == 1):# Plot images differently if we have classes as outputs or a continouse value.
    
    plt.axis('off')
    plt.imshow(npinputs)

    plt.subplot(1, 3, 2)
    plt.title('Predicted Segmentation')
    plt.axis('off')
    plt.imshow(newout[0, :, :])
    
    plt.subplot(1, 3, 3)
    plt.title('Ground Truth')
    plt.axis('off')
    plt.imshow(npclasses)

  else:
    plt.imshow(npinputs)

    plt.subplot(1, 3, 2)
    plt.title('Predicted Segmentation')
    plt.axis('off')
    plt.imshow(pred_seg)
    plt.subplot(1, 3, 3)
    plt.title('Ground Truth')
    plt.axis('off')
    plt.imshow(true_seg)
    plt.show()
    

    plt.figure(figsize=(8, 13))
    gs = gridspec.GridSpec(5, 4)
    gs.update(wspace=0.025, hspace=0.0)
    
    for ii in range(nc-1):#print the prediction stregth for each class
      plt.subplot(gs[ii])
      plt.axis('off')
      plt.imshow(newout[ii+1, :, :],vmin=np.amin(newout[1:]), vmax=np.amax(newout[1:]))
  plt.show()


  # return out1

