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

import time
import os
import glob
import torch
import random
import numpy as np
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import precision_score
import warnings

from .loaders import *


class train_model(object):
    def __init__(self, model,device,train_loader,eval_loader,image_path,class_path,evalimage_path,evalclass_path,writer,checkpoint_path,weightsmat,ps,bands,dividers,use_amp = False,amp = None,targetfloat = False,label = 'unlabeled', epoch = 0,print_every = 1 , show_every =1, eval_every =1,save_every =10,asses_every=4):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.image_path = image_path
        self.class_path = class_path
        self.evalimage_path = evalimage_path
        self.evalclass_path = evalclass_path

        self.writer = writer #tensorboard Summary writer

        self.checkpoint_path = checkpoint_path # where to save the model
        self.weightsmat = weightsmat # 2d array to for combining class predictions
        self.ps = ps    # padsize (cropping) of network output
        
        self.bands = bands  # the band indexes to display in the previews
        self.dividers = dividers # values to divide the displayed bands by to make them look good (aim for 0-1, ie 255,255,255 is a good divider for RGB imagery)
        self.use_amp = use_amp # whether to apply Nvidia AMP mixed precision)
        self.amp = amp
        self.targetfloat = targetfloat #if the target is a float instead of discrete classes
        self.label = label # a string to recognize the saved model by
        self.print_every =print_every # how often (epochs) to print a text summary of the model
        self.eval_every =eval_every # how often  (epochs) to run the eval dataset
        self.show_every =show_every# how often  (epochs) to show an image
        self.save_every =save_every# how often  (epochs) to save the model
        self.asses_every = asses_every# how often  (every nth batch) to calculate jaccard/precision
        self.epoch = epoch



    def __call__(self, criterion, optimizer,model = None,num_epochs=1):
        #example call:    trainer(criterion, optimizer4, num_epochs=100)

        if model == None:
            model = self.model
            
        batch_size = len(self.train_loader)
        train_losses = []
        eval_losses = []

        end_epoch = self.epoch + num_epochs


        train_samples = len(os.listdir(self.image_path))
        eval_samples = len(os.listdir(self.evalimage_path))


        e = self.epoch+1
        
        for e in range(e,e + num_epochs):
            self.epoch = e
            temptrain_loss = 0
            tempeval_loss = 0

            tempjacc_loss = 0
            tempjacc_losswsum = 0
            tempjacc_losssum = 0

            nc = self.weightsmat.shape[0]-1

            tempprecision = np.zeros(nc)
            statcounter = 0


            print ('-'*15,'Epoch %d' % e, '-'*15)

            t12sum = 0# store toal time 
            t23sum = 0# store toal time 

            jaccard_loss = 0
            jaccard_wloss = 0

            model.train()
            batch = 0

            for inputs,labels in tqdm(self.train_loader):
                batch +=1
                t1 = time.time()

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                if nc == 1:
                         labels = labels.unsqueeze(dim=1)          
                out = model(inputs.float())
                                
                if self.targetfloat:
                    loss = criterion(out, labels.float())
                else:
                    loss = criterion(out, labels.long())

                if self.use_amp:
                    with self.amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                optimizer.step()
                
                temptrain_loss += loss.item()
                train_loss =temptrain_loss/batch

                t2 = time.time()

                if(self.asses_every > 0):
                    if( batch % self.asses_every == 0):# if we need to asses accuracy in this batch

                        statcounter +=1
                        out2 = out.cpu().detach().numpy()
                        b_ =out2.argmax(1)   # get the class with the highest likelyhood.   
                        pred = b_.reshape(-1)
                        lbl = labels.cpu().numpy().reshape(-1)
                        
                        
                        zeroidxs = np.where(lbl==0)#ignore areas with a label value of zero in our accuracy calcs
                        lbl=np.delete(lbl, zeroidxs)
                        pred=np.delete(pred, zeroidxs)
                        jlossweight=len(pred)#get the weight of this batch (number of pixels) (to avoid a batch with a verly low number of pixels affecting our average)
                        jaccard = jsc(pred,lbl,average='micro')

                        jsum = jaccard*jlossweight

                        tempjacc_loss += jaccard
                        tempjacc_losssum += jsum
                        tempjacc_losswsum += jlossweight


                        jaccard_loss =tempjacc_loss/statcounter
                        jaccard_wloss =tempjacc_losssum/tempjacc_losswsum

         
                        unique,classcounts = np.unique(lbl, return_counts=True)
                        punique = np.unique(np.append(unique,np.unique(pred))) - 1
                        
                        warnings.filterwarnings('ignore')
                        # precs = np.array(precision_score(lbl, pred,labels=np.unique(pred), average=None))
                        precs = np.array(precision_score(lbl, pred, average=None))
                        warnings.filterwarnings('always')# often get a divide by 0 warning during jaccard

                        tempprecision[punique] += precs



                t3 = time.time()
                

                t12sum = t12sum + t2-t1
                t23sum = t23sum + t3-t2

                
            train_losses.append(train_loss)


            if (e-1) % self.show_every == 0:
                #alternate between training and eval images
                if e % 2 == 0:
                    print('training image')
                    show_image(self.model,self.device,self.weightsmat,self.train_loader,self.ps,self.bands,self.dividers)
                else:
                    print('eval image')
                    show_image(self.model,self.device,self.weightsmat,self.eval_loader,self.ps,self.bands,self.dividers)


            if (e-1) % self.eval_every == 0:
                #run evaulation dataset
                with torch.no_grad():
                    model.eval()

                    eval_loss = 0
                    evbatch = 0

                    for inputs,labels in tqdm(self.eval_loader):
                        evbatch += 1

                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        out = model(inputs.float())
                        loss = criterion(out, labels.long())

                        tempeval_loss += loss.item()
                        eval_loss = tempeval_loss/evbatch


                    eval_losses.append(eval_loss)

                    self.writer.add_scalar('Edata/Eval_Loss',eval_loss,e)            


            self.writer.add_scalar('Edata/Train_Loss',train_loss, e)
            self.writer.add_scalar('Edata/InvEJaccardw',1-jaccard_wloss,e)
            
            precision = tempprecision / statcounter#tempprecisionwsum
            precision[np.isinf(precision)] = 0


            og_dict = {}#Add all precisions to a single scaler in tensorboard
            for cl in range(nc):
                og_dict.update( {'class-'+str(cl+1) :1-precision[cl]} )   
            self.writer.add_scalars('precision/classes', og_dict, e)


            #print text summary
            if (e-1) % self.print_every == 0:
                print ('Epoch {}/{}...'.format(e, end_epoch),
                        'Tr_Loss {:6f}'.format(train_loss),
                        'Ev_Loss {:6f}'.format(eval_loss),

                        'Jaccard {:6f}'.format(jaccard_loss),
                        'Jaccardw {:6f}'.format(jaccard_wloss))
                print('precision:',precision)

                print('modeltime:',t12sum)
                print('assestime:',t23sum)

            self.writer.close()

            
            #Save Model
            if e % self.save_every == 0:
                checkpoint = {
                    'epochs' : e,
                    'state_dict' : self.model.state_dict(),
                    'opt_state_dict':optimizer.state_dict()
                }
                if(self.eval_every > 0):#if we are doing eval (TODO loss may be 0 if we dont eval on the same batch we save)
                    torch.save(checkpoint, self.checkpoint_path+'Unet-'+self.label+'-{}-Ev_Loss-{:2f}.pth'.format(e, eval_loss))
                else:
                    torch.save(checkpoint, self.checkpoint_path+'Unet-'+self.label+'-{}-jlos-{:2f}.pth'.format(e, jaccard_wloss))

                print ('Model saved!')
