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
import torch
import numpy as np
import shutil
import glob
from tqdm import tqdm_notebook as tqdm
from osgeo import gdal
import tifffile as tiff
import torch.utils.data as data

from .utils import *
from .loaders import *


def notransform(img):
    return img

def generate_image2(model,device,loader,workdir,input_folder,out_folder,weightsmat,ps,average =False,output = 'raw',prob_layer=1,prob_bins = 8,divider = 1):
    '''
    outputtypes:
    raw = classified with class nums as tiff values.
    colour = classified with colours defined by class.
    rawfloat = the averaged value output directly from the network. (select a clss to output) Can have very large file sizes
    prob = the averaged value runt run though a sigmoid and scaled to fit into a byte, (select a class to output,prob_bins and ,divider)

    '''

    infolder = workdir+input_folder
    outfolder = workdir+out_folder

    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
        print("Directory " , outfolder ,  " Created ")
    else:    
        shutil.rmtree(outfolder)    
        os.mkdir(outfolder)
        print("Directory " , outfolder ,  " was Cleaned")
    

    #using a dataloader, generate a tile for each image
    for img, filename in tqdm(loader):
        fname ="".join(filename)
        out= outputfile2(model, device,img,weightsmat,output,average,prob_layer,prob_bins,divider)
        if output == 'rawfloat':
            tiff.imwrite(outfolder+fname, out.astype('float'))
        else:
            tiff.imwrite(outfolder+fname, out.astype('uint8'))
        # break

    imlistsrc = glob.glob(infolder+'*'+'.tif')
    imlistdst = glob.glob(outfolder+'*'+'.tif')

    #get the coodinates of the source image offset by padsize and set output image coordinates

    for i in range(len(imlistsrc)):
        src = gdal.Open(imlistsrc[i])
        proj = src.GetProjectionRef()
        gt = src.GetGeoTransform()
        src = None

        gtl = list(gt)
        gtl[0] += ps * gtl[1] #multiply pixel shift by pixel size to get meters
        gtl[3] += ps * gtl[5] #TODO  do we need to incorporate  x and y portion of pixel size? (rotated images)

        dst = gdal.Open(imlistdst[i],gdal.GA_Update)
        out1 = dst.SetProjection(proj)
        out2 = dst.SetGeoTransform(tuple(gtl))
        dst = None


#takes an image from the dataloader and reurns a numpy array to save
def outputfile2(model,device,img,weightsmat,output = 'raw',average=False,prob_layer = 1,prob_bins = 8,divider = 1):

    model.eval()
    transformlist = [0]
    if(average):
        transformlist = range(8)

    for tr in transformlist:# do the folowing 8 times averaging the outputs
        trimg = all_transform(np.squeeze(img.numpy()),tr)
        tmg = torch.tensor(trimg.copy()).float()

        if len(tmg.size()) == 2:    #if single band 
            tmg = tmg.unsqueeze(dim = 2)

        tmg = tmg.transpose(0, 2)
        tmg = tmg.to(device)
        model.eval()
        out1 = model(tmg.unsqueeze(dim =0)).squeeze(dim =0)

        if output == 'prob': #make sure probabilitys are between 0 and 1
            out1 = torch.div(out1,divider)
            out1 = torch.sigmoid(out1)


        out2 = out1.cpu().detach().numpy()
        tempout = np.swapaxes(out2,2,0)

        out3 = all_untransform(tempout,tr)
        out3 = np.swapaxes(out3,2,0)
        out3 = np.swapaxes(out3,1,2)
     
        if tr == 0:
            outavg = out3
        else:
            outavg += out3

    outavg /= len(transformlist)


    model.train()

    temp = np.swapaxes(outavg,2,0)
    dotprod = np.dot( temp, np.asarray(weightsmat))

    if len(dotprod.shape) > 2:# calc the dotproduct of the output values and our weights matrix.
        newout = np.swapaxes(dotprod,0,2)
    else:
        newout = outavg
   
    b_ =newout.argmax(0)

    #return differnet forms of the output depending on settings
    if output == 'raw' :
        return b_
    elif output == 'colour':
        return decode_segmap(b_)
    elif output == 'prob':
        output = newout[int(prob_layer)]
        output *= prob_bins
        output = np.rint(output)
        return output.astype(np.uint8)
    elif output == 'rawfloat':
        output = newout[int(prob_layer)]
        return output.astype(np.float)
    else:
        print('bad output type')



def merge(infolder,outfolder,filename):
#merge all tiffs in the directory
    if os.path.isfile(outfolder+filename):
        os.remove(outfolder+filename)
    command = 'python /content/gdal_merge.py -o '+outfolder+filename +' '+ infolder+'*.tif'+ ' -of GTiff -co COMPRESS=DEFLATE -co BIGTIFF=YES'
    command = 'python /content/gdal_merge.py -o '+outfolder+filename +' '+ infolder+'*.tif'+ ' -of GTiff -co COMPRESS=DEFLATE'# dont need bigtiff in most cases
    return command

    
def makevrt(infolder,outfolder,filename):
#make a vrt of all tiffs in the directory (may be faster)
    imlist = glob.glob(infolder+'*.tif')
    
    file = open(infolder+'imlist.txt','w')  
    for item in imlist:
        file.write("%s\n" % item)
    file.close() 

    if os.path.isfile(outfolder+filename):
        os.remove(outfolder+filename)

    command =  'python /content/gdalbuildvrt.py" -allow_projection_difference' '-overwrite' '-input_file_list' + infolder +'imlist.txt ', +outfolder+filename+'.vrt' 

    return command

    
def mergevrt(infolder,outfolder,filename):
#convert the vrt into a tif
    command = 'python /content/gdal_merge.py -o '+outfolder+filename +' '+outfolder+filename+'.vrt'+ ' -of GTiff -co COMPRESS=DEFLATE '

    return command