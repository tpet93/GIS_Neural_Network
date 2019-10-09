'''
MIT License

Copyright (c) 2018 Joris
Modified by Tony Peter 2019

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

#Adapted From https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py by  Tony Peter 2019


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F


def actf(activation = 'Relu',num_parameters = 1):

    actf1 = nn.ReLU()
    if activation == 'Relu':
        actf1 = nn.ReLU()
    elif activation == 'Tanh':
        actf1 = nn.Tanh()
    elif activation == 'ELU':
        actf1 = nn.ELU()
    elif activation == 'LeakyReLU':
        actf1 = nn.LeakyReLU()
    elif activation == 'Tanhshrink':
        actf1 = nn.Tanhshrink()
    elif activation == 'Softplus':
        actf1 = nn.Softplus()
    elif activation == 'PReLU':
        actf1 = nn.PReLU(num_parameters= num_parameters)

    return actf1 

class UNetB(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        activation = 'Relu',
        up_mode='upconv',
        max_filters = 1024
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNetB, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
   
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, min(2 ** (wf + i),max_filters), padding, batch_norm,activation,max_filters)
            )
            prev_channels = min(2 ** (wf + i),max_filters)
            print(prev_channels)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, min(2 ** (wf + i),max_filters), up_mode, padding, batch_norm,activation,max_filters)
            )
            prev_channels = min(2 ** (wf + i),max_filters)
            print(prev_channels)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)
        
    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        x = self.last(x)
        return x

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm,activation,max_filters):
        super(UNetConvBlock, self).__init__()
        block = []
        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(actf(activation,out_size))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size,momentum = 0.005,track_running_stats=True))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(actf(activation,out_size))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size,momentum = 0.005,track_running_stats=True))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm,activation,max_filters):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2),
                actf(activation,out_size)
            )
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2,align_corners=False),
                nn.Conv2d(in_size, out_size, kernel_size=1),
                actf(activation,out_size)
            )

        if in_size == max_filters and out_size == max_filters:
            in_size = in_size*2
        # print('insize',in_size)
        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm,activation,max_filters)
        

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        # print(out.shape)
        out = self.conv_block(out)
        return out