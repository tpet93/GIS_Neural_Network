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

import sys
import os
import glob
import math
import h5py
import shutil
import random
# import tifffile as tiff
import skimage.external.tifffile as tiff
import h5py

import numpy as np

from osgeo import gdal
from osgeo import ogr
from osgeo import gdalconst

from tqdm import tqdm_notebook as tqdm


class Settingsclass:
    maxshift = 0
    ps = 0
    shiftspertile = 1
    class_folder = ""
    color_folder = ""
    output_filename = ""
    area = 1
    virtualtilesize = 0
    nullthresh = 0
    tile_size_y = 0
    tile_size_x = 0
    countall = False

def progress_callback(complete, message, unknown):
    print('progress: {}'.format(complete))
    return 1

def rasterizegpkg(refraster,outputfolder,gpkgfile,attribute,filename):

    reference = gdal.Open(refraster, gdalconst.GA_ReadOnly)
    # print('openreferernce')

    geo_transform = reference.GetGeoTransform()
    referenceProj = reference.GetProjection()

    x = reference.RasterXSize 
    y = reference.RasterYSize


    mb_v = ogr.Open(gpkgfile)
    mb_l = mb_v.GetLayer()
    
    output = outputfolder+filename+'.tif'

    driver = gdal.GetDriverByName('GTiff')
    # target_ds = driver.Create(output, x, y, 1,gdal.GDT_Float32)
    # print('create driver')

    target_ds = driver.Create(output, x , y , 1 , gdal.GDT_Byte)

    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(referenceProj)

    band = target_ds.GetRasterBand(1)

    band.FlushCache()

    gdal.RasterizeLayer(target_ds, [1], mb_l, options=["ATTRIBUTE="+attribute,'COMPRESS=DEFLATE'],callback = progress_callback)

    band = None
    target_ds = None
    mb_v = None
    mb_l = None

def gen_trainingtilesfromgpkg(colourfile,classfile,attribute,workdir,color_folder,class_folder,
tilesize,maxshift = 0,shiftspertile = 1,ps = 0,nullthresh = 0.3,countall = False):#this method and some of the below ones tile by clipping a memmaped numpy array, this is very fast but loses goespatial information.


    color_folder = workdir+color_folder+'/'
    class_folder =  workdir+class_folder+'/'

    output_filename = 'tile_'
    blackcounter = 0
    imgcounter = 0

    if  os.path.exists(color_folder):
        shutil.rmtree(color_folder)
    if  os.path.exists(class_folder):
        shutil.rmtree(class_folder)


    os.mkdir(color_folder)
    os.mkdir(class_folder)


    tile_size_x = tilesize
    tile_size_y = tilesize
    ps = int(ps)

    virtualtilesize = tile_size_x - ps*2
    # pool = mp.Pool(1)

    print('Rasterizing Vector Layer')
    rasterizegpkg(colourfile,workdir,classfile,attribute,attribute+'_rasterized')

    # dscol = gdal.Open(input_folder+colorfiles[area]+".tif")
    # dsclas = gdal.Open(input_folder+classfiles[area]+".tif")

    # f_image= workdir + 'col.npy'
    # f_class= workdir + 'class.npy'

    # # MEMMAP allows you to map to a numpy array directly on disk instead of storing it in memory
    # dscol= np.memmap(f_image, dtype='float', mode='w+')
    # dsclas = np.memmap(f_class, dtype='int', mode='w+')

   

    # dscol = tiff.imread(colourfile,out = 'memmap')
    # print('doneread')
    # tiff.imsave(colourfile+'2.tif',dscol)

    # with tiff.TiffFile(colourfile) as tif:
    #     dscol = tif.asarray(out = 'memmap')
    #     print(dscol.shape)
    # print(dscol.shape)
    # print('Readclass')

    # with tiff.TiffFile(workdir+attribute+'_rasterized.tif') as tif:
    #     dsclas = tif.asarray(out = 'memmap')
    #     print(dsclas.shape)
    # print(dsclas.shape)


    with tiff.TiffFile(colourfile) as tif:
        dscol = tif.asarray(memmap=True)
        print('Raster Resolution:',dscol.shape)

    with tiff.TiffFile(workdir+attribute+'_rasterized.tif') as tif:
        dsclas = tif.asarray(memmap=True)

    xsize = dscol.shape[0]
    ysize = dscol.shape[1]

    xtiles = math.ceil(xsize/virtualtilesize)
    ytiles = math.ceil(ysize/virtualtilesize)

    # pbar = tqdm(total=math.ceil(xsize/virtualtilesize))
    tps = math.ceil(virtualtilesize+maxshift)

    settings = Settingsclass()

    settings.maxshift = maxshift
    settings.ps = ps
    settings.shiftspertile = shiftspertile
    settings.class_folder = class_folder
    settings.color_folder = color_folder
    settings.output_filename = output_filename
    settings.virtualtilesize = virtualtilesize
    settings.nullthresh = nullthresh
    settings.tile_size_y = tile_size_y
    settings.tile_size_x = tile_size_x
    settings.countall = countall
    settings.tps = tps


    # dscol[:] = np.pad(dscol, ((tps, tps),(tps, tps),(0,0)), 'constant')[:]
    # dsclas[:] = np.pad(dsclas, ((tps, tps),(tps, tps)), 'constant')[:]

    
    print("Tiling...")
 
    pbar = tqdm(total=xtiles*ytiles)
    for i in range(0, xsize, virtualtilesize): 
        for j in range(0, ysize, virtualtilesize):
            ims,blacks = worker(i,j, dscol, dsclas,settings)
            blackcounter += blacks
            imgcounter += ims
            pbar.update(1)

            
    pbar.close()


    print(imgcounter," tiles generated")
    print(blackcounter," tiles with less than ",nullthresh*100,'% classified removed')

def worker(i,j, dscol, dsclas, settings,ds_te_clas=None):
    # print(i,j)
    pixels = []
    ps = settings.ps
    virtualtilesize = settings.virtualtilesize
    blackcounter = 0
    imgcounter = 0

    for shifti in range(settings.shiftspertile):
        xoffset = random.randint(-settings.maxshift,settings.maxshift)
        yoffset = random.randint(-settings.maxshift,settings.maxshift)
        minx = settings.tps+i+xoffset
        maxx = settings.tps+i+xoffset+settings.tile_size_x
        miny = settings.tps+j+yoffset
        maxy = settings.tps+j+yoffset+settings.tile_size_y

        black = True #default
        minx = max(minx,0)
        maxx = min(maxx,dscol.shape[0])        
        miny = max(miny,0)
        maxy = min(maxy,dscol.shape[1])


        # dscol[:] = np.pad(dscol, ((tps, tps),(tps, tps),(0,0)), 'constant')[:]
        # dsclas[:] = np.pad(dsclas, ((tps, tps),(tps, tps)), 'constant')[:]

        dopad = False
        xpad = 0
        ypad = 0

        crclass = dsclas[minx:maxx,miny:maxy]
        if (crclass.shape[0] != settings.tile_size_x)  or  (crclass.shape[1] != settings.tile_size_y):
            dopad = True
            xpad = int(settings.tile_size_x-crclass.shape[0])
            ypad = int(settings.tile_size_y-crclass.shape[1])
            
            crclass = np.pad(crclass, ((int(xpad/2), xpad-(int(xpad/2))),(int(ypad/2), ypad-int(ypad/2))), 'constant')


        croppedclass = crclass[ps:-ps,ps:-ps]
        
        # print(croppedclass.shape)
        # count = np.count_nonzero(croppedclass)
        # print(count)



        if settings.countall:
            count = np.count_nonzero(croppedclass)
            black = count<(virtualtilesize*virtualtilesize*settings.nullthresh)

        else:
            vcount = np.count_nonzero(np.count_nonzero(croppedclass, axis=0))
            hcount = np.count_nonzero(np.count_nonzero(croppedclass, axis=1))
            avg = vcount+hcount/2
            black = avg<(virtualtilesize*settings.nullthresh)

        if(black):
            blackcounter += 1
        else:

            crcol = dscol[minx:maxx,miny:maxy]
            if dopad:
                crcol = np.pad(crcol, ((int(xpad/2), xpad-(int(xpad/2))),(int(ypad/2), ypad-int(ypad/2)),(0,0)), 'constant')

            colfname = settings.color_folder+settings.output_filename+ str(settings.area)+'_' + f'{minx:06}' + "_" + f'{miny:06}' + ".tif"
            classfname = settings.class_folder+settings.output_filename + str(settings.area)+'_' + f'{minx:06}' + "_" + f'{miny:06}' + ".tif"


            tiff.imsave(colfname, crcol )
            tiff.imsave(classfname, crclass )
            imgcounter += 1
        
    return(imgcounter,blackcounter)



def gen_trainingtiles2(input_folder,colorfiles,classfiles,workdir,color_folder,class_folder,tilesize,
maxshift = 0,shiftspertile = 1,ps = 0,nullthresh = 0.3,countall = False):
    

    color_folder = workdir+color_folder+'/'
    class_folder =  workdir+class_folder+'/'

    output_filename = 'tile_'
    blackcounter = 0
    imgcounter = 0

    if  os.path.exists(color_folder):
        shutil.rmtree(color_folder)
    if  os.path.exists(class_folder):
        shutil.rmtree(class_folder)

    os.mkdir(color_folder)
    os.mkdir(class_folder)

    tile_size_x = tilesize
    tile_size_y = tilesize
    ps = int(ps)

    virtualtilesize = tile_size_x - ps*2
    # pool = mp.Pool(1)

    for area in range(len(colorfiles)):

        # dscol = gdal.Open(input_folder+colorfiles[area]+".tif")
        # dsclas = gdal.Open(input_folder+classfiles[area]+".tif")

        dscol = tiff.imread(input_folder+colorfiles[area]+".tif")
        dsclas = tiff.imread(input_folder+classfiles[area]+".tif")

        xsize = dscol.shape[0]
        ysize = dscol.shape[1]

        xtiles = math.ceil(xsize/virtualtilesize)
        ytiles = math.ceil(ysize/virtualtilesize)
        pbar = tqdm(total=xtiles*ytiles)
        # pbar = tqdm(total=math.ceil(xsize/virtualtilesize))
        tps = math.ceil(virtualtilesize+maxshift)
  
        settings = Settingsclass()

        settings.maxshift = maxshift
        settings.ps = ps
        settings.shiftspertile = shiftspertile
        settings.class_folder = class_folder
        settings.color_folder = color_folder
        settings.output_filename = output_filename
        settings.area = area
        settings.virtualtilesize = virtualtilesize
        settings.nullthresh = nullthresh
        settings.tile_size_y = tile_size_y
        settings.tile_size_x = tile_size_x
        settings.countall = countall
        settings.tps = tps

        dscol = np.pad(dscol, ((tps, tps),(tps, tps),(0,0)), 'constant')
        dsclas = np.pad(dsclas, ((tps, tps),(tps, tps)), 'constant')

        
        for i in range(0, xsize, virtualtilesize): 
            for j in range(0, ysize, virtualtilesize):
                ims,blacks = worker(i,j, dscol, dsclas,settings)
                blackcounter += blacks
                imgcounter += ims
                pbar.update(1)

             
    
        print('Done with: ',colorfiles[area])
        pbar.close()


    print(imgcounter," tiles generated")
    print(blackcounter," tiles with less than ",nullthresh*100,'% classified removed')


def gen_trainingtiles(input_folder,colorfiles,classfiles,workdir,color_folder,class_folder,tilesize,maxshift = 0,shiftspertile = 1,ps = 0,nullthresh = 0.3,countall = False):
    

    color_folder = workdir+color_folder+'/'
    class_folder =  workdir+class_folder+'/'

    output_filename = 'tile_'
    blackcounter = 0
    imgcounter = 0

    if  os.path.exists(color_folder):
        shutil.rmtree(color_folder)
    if  os.path.exists(class_folder):
        shutil.rmtree(class_folder)

    os.mkdir(color_folder)
    os.mkdir(class_folder)

    tile_size_x = tilesize
    tile_size_y = tilesize
    ps = int(ps)

    virtualtilesize = tile_size_x - ps*2

    for area in range(len(colorfiles)):

        dscol = gdal.Open(input_folder+colorfiles[area]+".tif")
        dsclas = gdal.Open(input_folder+classfiles[area]+".tif")

        band = dscol.GetRasterBand(1)
        xsize = band.XSize
        ysize = band.YSize
        xtiles = math.ceil(xsize/virtualtilesize)
        ytiles = math.ceil(ysize/virtualtilesize)
        pbar = tqdm(total=xtiles*ytiles)
        # pbar = tqdm(total=math.ceil(xsize/virtualtilesize))

        for i in range(0, xsize, virtualtilesize): 
            for j in range(0, ysize, virtualtilesize): 
                pixels = []
                for shifti in range(shiftspertile):
                    xoffset = random.randint(-maxshift,maxshift)
                    yoffset = random.randint(-maxshift,maxshift)

                    pixels = [i+xoffset,j+yoffset,tile_size_x,tile_size_y] 

                    gdal.Translate(class_folder+output_filename+ str(area)+'_' + str(pixels[0]) + "_" + str(pixels[1]) + ".tif", dsclas,format = 'GTiff',srcWin = pixels)#do class
                    gdal.Translate(color_folder+output_filename + str(area)+'_' + str(pixels[0]) + "_" + str(pixels[1]) + ".tif", dscol,format = 'GTiff',srcWin = pixels) #do colour

                    im = tiff.imread(class_folder+output_filename + str(area)+'_' + str(pixels[0]) + "_" + str(pixels[1]) + ".tif")
                    cropimg = im[ps:-ps,ps:-ps]
                    vcount = np.count_nonzero(np.count_nonzero(cropimg, axis=0))
                    hcount = np.count_nonzero(np.count_nonzero(cropimg, axis=1))
                    avg = vcount+hcount/2
                    count = np.count_nonzero(cropimg)
                    black = True
                    if countall:
                        black = count<(virtualtilesize*virtualtilesize*nullthresh)
                    else:
                        black = avg<(virtualtilesize*nullthresh)

                    if(black):
                        blackcounter += 1
                        os.remove(color_folder+output_filename + str(area)+'_' + str(pixels[0]) + "_" + str(pixels[1]) + ".tif") # if black remove colour
                        os.remove(class_folder+output_filename + str(area)+'_' + str(pixels[0]) + "_" + str(pixels[1]) + ".tif") # if black remove class
                    else:
                        imgcounter +=1

                pbar.update(1)

        print('Done with: ',colorfiles[area])
        pbar.close()
            
    for file in glob.glob(class_folder+"*.xml"):
      os.remove(file)
    for file in glob.glob(color_folder+"*.xml"):
      os.remove(file)

    print(imgcounter," tiles generated")
    print(blackcounter," tile with less than ",nullthresh*100,'% classified removed')


def gen_fulltiles(input_folder,colorfile,workdir,output_folder,tilesize,ps = 0):#this method uses the slower gdal translate to genereate the tile as this preserves the geospatial information allowing us to merge the tiles afterwards

    if  os.path.exists(workdir+output_folder):
        shutil.rmtree(workdir+output_folder)
    os.mkdir(workdir+output_folder)

    ps = int(ps)

    dscolfull = gdal.Open(input_folder+colorfile)

    band = dscolfull.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize
    xtiles = math.ceil(xsize/(tilesize-ps*2))
    ytiles = math.ceil(ysize/(tilesize-ps*2))
    pbar = tqdm(total=xtiles*ytiles)
    output_filename = 'tile_'

    tile_size_x = tilesize
    tile_size_y = tilesize
  
    virtualtilesize = tile_size_x - ps*2

    for i in range(0, xsize, virtualtilesize):
        for j in range(0, ysize,virtualtilesize):        
            pixels = [i-ps,j-ps,tile_size_x,tile_size_y] # offset image by padsize so the coordinate of the topleft corner is correct
            gdal.Translate(workdir+output_folder+output_filename + str(i) + "_" + str(j) + ".tif", dscolfull,format = 'GTiff',srcWin = pixels)#do full
            pbar.update(1)

    pbar.close()
    

def split_shuffle(color_input_folder,class_input_folder,ps = 0,nullthresh = 0.1):
#Random image join

    ps = int(ps)
    filenames = [os.path.basename(x) for x in glob.glob(color_input_folder+"tile_*.tif")]
  
    inp_files = list(map(lambda x : x.split('.')[0] + '.tif', filenames))
    total_files =len(inp_files)
    all_idxs = np.array(range(total_files))
    random.shuffle(all_idxs)
#     print(all_idxs)
    blackcounter = 0
    imgcounter = 0
    for i in tqdm(all_idxs):
    #   print(i)
        filename1= inp_files[all_idxs[i]]
        filename2= inp_files[all_idxs[(i+1)%total_files]]

        img1 = tiff.imread(color_input_folder + filename1.split('.')[0] + '.tif')
        smg1 = tiff.imread(class_input_folder + filename1.split('.')[0] + '.tif')
        
        imwidth = (img1.shape[0])
        hw=int(imwidth/2)
        
        img1s = np.array(img1[:,0:hw,:])
        smg1s = np.array(smg1[:,0:hw])
        
        smg2s = np.zeros(smg1s.shape)
        cropsmg = smg2s[ps:-ps,:-ps]     
        
        tilesize = smg1s.shape[0] - ps*2 #segmented image is smaller than color
#         print(tilesize-)
        cropimg = smg1s[ps:-ps,ps:]        
#         print(cropimg.shape)


        
        if(cropimg.any(axis=-1).sum()>(tilesize*nullthresh)):
            attempts = 0
            while (cropsmg.any(axis=-1).sum()<(tilesize*nullthresh)) and (attempts < 10):    
                filename2= inp_files[all_idxs[random.randint(0,total_files-1)]]       
                img2 = tiff.imread(color_input_folder + filename2.split('.')[0] + '.tif')
                smg2 = tiff.imread(class_input_folder + filename2.split('.')[0] + '.tif')
                img2s = np.array(img2[:,0:hw,:])
                smg2s = np.array(smg2[:,0:hw])
                cropsmg = smg2s[ps:-ps,:-ps]        
                           

#                 if (attempts > 0) :
#                    print('retry',attempts)
                attempts +=1



            if(smg2s.any(axis=-1).sum()>(tilesize*nullthresh)):
                                     
                outimg=np.concatenate([img1s,img2s],1)
                outsimg=np.concatenate([smg1s,smg2s],1)
                
#                 cropimg = outsimg[ps:-ps,ps:-ps]                            
                tiff.imsave(color_input_folder+'split_'+filename1, outimg)
                tiff.imsave(class_input_folder+'split_'+filename1, outsimg)
                imgcounter += 1


        else:
            blackcounter += 1

    #   break
    print(imgcounter," tiles generated")
    print(blackcounter," images with less than ",nullthresh*100,'% classified removed')




def makehdf5(datasetpath,input_path, segmented_path,ps,bandtransform =None,lbltransform = None,targettype = 'i8'):
    ps = int(ps)
    if not os.path.exists(datasetpath):
        os.mkdir(datasetpath)
    if os.path.isfile(datasetpath+'dataset.hdf5'):
        os.remove(datasetpath+'dataset.hdf5')
            
    f = h5py.File(datasetpath+'dataset.hdf5', "w")
    
    filenames = [os.path.basename(x) for x in glob.glob(input_path+"*.tif")]
    total_files = len(filenames)
    
    for ii in tqdm(range(total_files)):
        filename= filenames[ii]

        img = tiff.imread(input_path + filename)
        if bandtransform is not None:
            img = bandtransform(img)


        singleband = len(img.shape) == 2
        if(singleband):
            img = np.expand_dims(img, axis=2)

        imgl = tiff.imread(segmented_path + filename)
        imgl = imgl[ps:-ps,ps:-ps]

        if lbltransform is not None:
            imgl = lbltransform(imgl)

        if ii == 0:
            images = f.create_dataset("images", (total_files,)+img.shape, dtype='f')
            labels = f.create_dataset("labels",(total_files,)+imgl.shape, dtype=targettype)#i8
        
        images[ii] = img
        labels[ii] = imgl
    f.close()