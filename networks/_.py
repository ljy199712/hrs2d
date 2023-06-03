from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from hr_layers import *
from layers import upsample

class HRDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, mobile_encoder=False):
        super(HRDepthDecoder, self).__init__()
        self.num_output_channels = num_output_channels  #1
        self.num_ch_enc = num_ch_enc   #[ 64 , 18 , 36 , 72 ,144 ]
        self.scales = scales    #[ 0 , 1 , 2 , 3 ]
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])# 解码器通道[ 16 ，32 ，64 ，128 ，256 ]
        self.convs = nn.ModuleDict()
        self.norm_layer = nn.BatchNorm2d
        self.convs1 = nn.Conv2d(256,128, #32   通道变为两倍
                                  3,
                                  1,
                                  1,
                                  bias=False)
                        
                        # nn.ReLU(inplace=True)
        # decoder
        self.convs = nn.ModuleDict()
        # ModuleDict可以像常规Python字典一样索引，同样自动将每个 module 的 parameters 添加到网络之中的容器(

        
#         class ModuleDict(nn.Module):
#       def __init__(self):
#         super(ModuleDict, self).__init__()
#         self.choices = nn.ModuleDict({
#             'conv': nn.Conv2d(10, 10, 3),  选择是一个输入输出通道为10，滤波器为3的卷积  和一个 尺寸为3*3的最大池化
#             'pool': nn.MaxPool2d(3)        
#         })                                  完成后尺寸变为 n/2 - 1  输出通道为10

#         self.activations = nn.ModuleDict({
#             'relu': nn.ReLU(),
#             'prelu': nn.PReLU()
#         })                                 #激活函数为relu

#      def forward(self, x, choice, act):        
#         x = self.choices[choice](x)
#         x = self.activations[act](x)
#         return x



        
        # 适应块
        if self.num_ch_dec[0] < 16:   #self.num_ch_dec[0]  = 64
            self.convs["up_x9_01"] = ConvBlock(self.num_ch_enc[4],self.num_ch_enc[3] * 2)  # 卷积块 通道修改 由 前面的到后面的 

            self.convs["up_x9_0"] = ConvBlock(self.num_ch_dec[1],self.num_ch_dec[0])#（ 18 ， 64） 通道由18变为了 64
            self.convs["up_x9_1"] = ConvBlock(self.num_ch_dec[0],self.num_ch_dec[0])#（ 18 ， 18） 
        
        # adaptive block
            self.convs["72"] = Attention_Module(2 * self.num_ch_dec[4],  2 * self.num_ch_dec[4]  , self.num_ch_dec[4]) #（ 512 ， 512 ， 256 ） high low  输入是512+512 = 1024 输出是256
            self.convs["36"] = Attention_Module(self.num_ch_dec[4], 3 * self.num_ch_dec[3], self.num_ch_dec[3]) # （256 ， 384 ， 128）         high low  输入是256+284 = 540 输出是128
            self.convs["18"] = Attention_Module(self.num_ch_dec[3], self.num_ch_dec[2] * 3 + 64 , self.num_ch_dec[2])#（ 128 ，256 ，64）       high low  输入是128+256 = 384 输出是64
            self.convs["9"] = Attention_Module(self.num_ch_dec[2], 64, self.num_ch_dec[1]) #（64 ， 64 ， 32 ）                                 high low  输入是64+64 = 128 输出是32
        else:      
                                                                               # [ 16 ，32 ，64 ，128 ，256 ]
            self.convs["up_x9_0"] = ConvBlock(self.num_ch_dec[1],self.num_ch_dec[0])#（ 32 ， 16 ）                        high low  输入是32+16 = 32 输出是32
            self.convs["up_x9_1"] = ConvBlock(self.num_ch_dec[0],self.num_ch_dec[0])#（ 16 ， 16 ）                        high low  输入是16+16 = 32 输出是16
            self.convs["72"] = Attention_Module(self.num_ch_enc[4]  , self.num_ch_enc[2] * 3, 256) #（ 256 ，256 ， 256）  high low  输入是256+256 = 512  输出是256
            self.convs["36"] = Attention_Module(256, self.num_ch_enc[2] * 3, 128) #（ 256 ， 192 ， 128 ）                 high low  输入是256+192 = 448  输出是128
            self.convs["18"] = Attention_Module(128, self.num_ch_enc[1] * 3 + 64 , 64) # （ 128 ， 160 ， 64 ）            high low  输入是128+160 = 288  输出是64
            self.convs["9"] = Attention_Module(64, 64, 32)#（ 64 64 32 ）                                                  high low  输入是64+64 = 128  输出是32
        for i in range(4):
            self.convs["dispConvScale{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))

        # nn.ModuleList里面储存了不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器(注册)，里面的module是按照List的形式顺序存储的，但是在forward中调用的时候可以随意组合。
        # 可以任意将 nn.Module 的子类 (比如 nn.Conv2d, nn.Linear 之类的) 加到这个 list 里面，方法和 Python 自带的 list 一样，也就是说它可以使用 extend，append 等操作。

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        outputs = {}                   # [ 64 , 18 , 36 , 72 ,144 ]
        feature144 = input_features[4]
        feature72 = input_features[3]
        feature36 = input_features[2]
        feature18 = input_features[1]
        feature64 = input_features[0]


        x72 = self.convs["72"](feature144, feature72)
        x36 = self.convs["36"](x72 , feature36)
        x18 = self.convs["18"](x36 , feature18)
        x9 = self.convs["9"](x18,[feature64])
        x6 = self.convs["up_x9_1"](upsample(self.convs["up_x9_0"](x9)))
        print(x72.shape,  x36.shape , x18.shape , x9.shape, x6.shape)

       
        outputs[("disp",0)] = self.sigmoid(self.convs["dispConvScale0"](x6))
        outputs[("disp",1)] = self.sigmoid(self.convs["dispConvScale1"](x9))
        outputs[("disp",2)] = self.sigmoid(self.convs["dispConvScale2"](x18))
        outputs[("disp",3)] = self.sigmoid(self.convs["dispConvScale3"](x36))
        return outputs
        
