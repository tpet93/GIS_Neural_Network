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

import numpy as np
import random
from tqdm import tqdm_notebook as tqdm

def decode_segmap(image): #colourize the Labels for use in display
  
    background = [128, 128, 128]
    primary = [0, 68, 27]
    sec = [158,216,152]
    sav = [204,252,247]
    isl = [209, 122, 209]
    ww = [29,134,65]
    stunt = [95,140,214]
    heath = [200,178,144]
    road = [0,0, 0]
    earth = [121, 73, 183]
    river = [169,0, 5]


    label_colours = np.array([background, primary, sec, sav, isl,ww, stunt, heath,road,earth,river]).astype(np.uint8)
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, 11):
        r[image == l] = label_colours[l, 0]
        g[image == l] = label_colours[l, 1]
        b[image == l] = label_colours[l, 2]

    rgb = np.zeros((image.shape[0], image.shape[1], 3)).astype(np.uint8)
    rgb[:, :, 0] = b #BGR is used instead of RGB
    rgb[:, :, 1] = g
    rgb[:, :, 2] = r
    return rgb

class RandomNoise(object): #unused add random noise to input images
    def __init__(self, probability = 0.5):
         self.probability = probability
    def __call__(self, img):
        if random.random() <= self.probability:
#           print(img.shape)
          noise = np.random.normal(1, .01, img.shape)#sd. 1% change
          img = np.multiply(img,noise)
#           print(img.shape)

        return img
      
      
def Randomtransform(img,limg):#randomly flip and or rotate both input and labels image to one of 8 possibilities

  if (random.randint(0,1) == 1):
     img=np.flip(img, axis=1)    
     limg=np.flip(limg, axis=1)
  
  randnum  = random.randint(0,3)
  img=np.rot90(img, randnum)
  limg=np.rot90(limg, randnum)
  return img,limg   

def all_transform(img,transform): # apply a transform (1 of 8) to an image defineied by input argument
    #used during final classification to run each image through 8 times.
    #0-3 rot no flip, 4-7, rotate and flip H
    if(transform>7) or (transform< 0):
        print ('Error: invalid transform')
    if(transform>3):
        img=np.flip(img, axis=1)
    rotnum  = transform % 4
    img=np.rot90(img, rotnum,axes=(0, 1))
    return img   

def all_untransform(img,transform):# apply the reverse of a transform (1 of 8) to an image defineied by input argument
    #used with above function to return the image to normal orientation before averaging.
    #0-3 rot no flip, 4-7, rotate and flip H
    if(transform>7) or (transform< 0):
        print ('Error: invalid transform')

    rotnum  = transform % 4
    img=np.rot90(img,4-rotnum,axes=(0, 1))

    if(transform>3):
        img=np.flip(img, axis=1)
    return img    




def get_class_weights(loader,num_classes,ps = 0):

    #counts the number of pixels of each class in our training loader.
    #returns the inverse of the proportions which is used as weights to account for class imbalance

    all_labels = np.zeros(num_classes, dtype=float) 
    for _, labels in tqdm(loader): 
        labels = np.array(labels)
        labels = labels.reshape(-1)
        unique, counts = np.unique(labels, return_counts=True)
        for i in range(len(unique)):
            classnum = unique[i]
            count = counts[i]
            all_labels[int(classnum)] += float(count)


    all_labels[0] = 1 #cant divide by zero yet
    all_labels = all_labels.max() /all_labels
    all_labels[np.isinf(all_labels)] = 1
    all_labels[0] = 0 #zeroth class is ignore, now we can divide by 0


    return all_labels

def calc_padsize(depth,shape):# calculates how much cropping is needed due to the valid padding of the network.
    outs = shape
    outs -= 4
    isneat = True
    for i in range(depth-1):
        if not float(outs/2).is_integer():
            isneat = False
        outs = int(outs/2)
        outs -= 4

    for i in range(depth-1):
        outs *=2
        outs -= 4
    return isneat,(shape - outs)/2




def find_neats(lowestres,depth):# calculates input resolutions that downscale cleanly, input isth e lowest resolution of the bottleneck
    x = lowestres
    for i in range(depth-1):
        x += 4
        x *=2
    x += 4
    return x