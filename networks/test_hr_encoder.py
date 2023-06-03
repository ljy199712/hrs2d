import os
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url
from torchsummary import summary

import matplotlib.pyplot as plt
logger = logging.getLogger('hrnet_backbone')

__all__ = ['hrnet18', 'hrnet32', 'hrnet48','hrnet64']
e =0

model_urls = {
    'hrnet18_imagenet': 'https://opr0mq.dm.files.1drv.com/y4mIoWpP2n-LUohHHANpC0jrOixm1FZgO2OsUtP2DwIozH5RsoYVyv_De5wDgR6XuQmirMV3C0AljLeB-zQXevfLlnQpcNeJlT9Q8LwNYDwh3TsECkMTWXCUn3vDGJWpCxQcQWKONr5VQWO1hLEKPeJbbSZ6tgbWwJHgHF7592HY7ilmGe39o5BhHz7P9QqMYLBts6V7QGoaKrr0PL3wvvR4w',
    'hrnet32_imagenet': 'https://opr74a.dm.files.1drv.com/y4mKOuRSNGQQlp6wm_a9bF-UEQwp6a10xFCLhm4bqjDu6aSNW9yhDRM7qyx0vK0WTh42gEaniUVm3h7pg0H-W0yJff5qQtoAX7Zze4vOsqjoIthp-FW3nlfMD0-gcJi8IiVrMWqVOw2N3MbCud6uQQrTaEAvAdNjtjMpym1JghN-F060rSQKmgtq5R-wJe185IyW4-_c5_ItbhYpCyLxdqdEQ',
    'hrnet48_imagenet': 'https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ',
    'hrnet48_cityscapes': 'https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ',
}

def visual_feature(features):
    for a in range(len(features)):
        feature_map = features[a].squeeze(0).cpu()
        n,h,w = feature_map.size()
        print("{} channel in stage {}".format(n,a))
        list_mean = []
        sum_feature_map = torch.sum(feature_map,0)
        #sum_feature_map,_ = torch.max(feature_map,0)
        for i in range(n):
            list_mean.append(torch.mean(feature_map[i]))
        
        sum_mean = sum(list_mean)
        feature_map_weighted = torch.ones([n,h,w])
        for i in range(n):
            feature_map_weighted[i,:,:] = (torch.mean(feature_map[i]) / sum_mean) * feature_map[i,:,:]
        sum_feature_map_weighted = torch.sum(feature_map_weighted,0)
        plt.imshow(sum_feature_map,cmap= 'magma')
        plt.savefig('feature_viz/{}_stage.png'.format(a))
        #plt.savefig('feature_viz_ori/{}_stage.png'.format(a))
        plt.imshow(sum_feature_map_weighted,cmap = 'magma')
        plt.savefig('feature_viz/{}_stage_weighted.png'.format(a))
        #plt.savefig('feature_viz_ori/{}_stage_weighted.png'.format(a))


################################         3x3的卷积快
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

################################        1x1的卷积快
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class simam_module(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()
 
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
 
    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s
 
    # staticmethod
    def get_module_name():
        return "simam"
 
    def forward(self, x):
 
        b, c, h, w = x.size()
        
        n = w * h - 1
 
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
 
        return x * self.activaton(y)

class ASPP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel,out_channel, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(out_channel * 5, out_channel, 1, 1)
        # self.attention = simam_module(out_channel * 5)
    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear', align_corners=True)

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):  

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None: #这里的downsample的作用是希望输入原图x与conv3输出的图维度相同，方便两种特征图进行相加，保留更多的信息（你要是看不懂这句话，就去先简单了解一下残差结构）
            #如果x与conv3输出图维度本来就相同，就意味着可以直接相加，那么downsample会为空，自然就不会进行下面操作

            identity = self.downsample(x)    #downsample = nn.Sequential(
    						             	 #        nn.Conv2d(64, 64*4,kernel_size=1, stride=1, bias=False),
    							             #        nn.BatchNorm2d(64*4, momentum=BN_MOMENTUM),


        out += identity   ##残差结构相加嘛
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    # (1) 首先判断是否降维或者输入输出的通道(num_inchannels[branch_index]    和    num_channels[branch_index] * block.expansion(通道扩张率))是否一致，不一致使用1z1卷积进行维度升/降，后接BN，不使用ReLU；
    # (2) 顺序搭建num_blocks[branch_index]个block，第一个block需要考虑是否降维的情况，所以单独拿出来，后面1 到 num_blocks[branch_index]个block完全一致，使用循环搭建就行。此时注意在执行完第一个block后将num_inchannels[branch_index重新赋值为 num_channels[branch_index] * block.expansion。

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True, norm_layer=None):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        # (1) 首先判断是否降维或者输入输出的通道(num_inchannels[branch_index]    和    num_channels[branch_index] * block.expansion(通道扩张率))是否一致，不一致使用1z1卷积进行维度升/降，后接BN，不使用ReLU；
        #  """如果输入输出的维度不一致的话，downsample就重新构建"""
         #这里与上面第二步的self._make_layer类似，也是一个残差结构
    #这里block.expansion为1，self.num_inchannels是[32,64],num_channels[32,64]所以就不用下采样改变通道数了
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],   #输入通道
                          num_channels[branch_index] * block.expansion,   #输出通道   即是否需要降维   这个函数可以完成升维和降维
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(num_channels[branch_index] * block.expansion),
            )
        layers = []
        
        #layers第一层为：
        """分支的第一个层，的输入输出维度是不同的所以单独构建"""
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample, norm_layer=self.norm_layer))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        
        #通道数依然是[32,64]
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index], norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

    #num_branch为2  ，

    
    #在stage1中branch的第一个元素为self._make_one_branch(0, BASICBLOCK, [4,4], [32,64])
    #第二个元素为：self._make_one_branch(1, BASICBLOCK, [4,4], [32,64])

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        #如果只有一个分支，则不需要融合
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches #2
        num_inchannels = self.num_inchannels #[32,64]
        fuse_layers = []
    #如果self.multi_scale_output为True，意味着只需要输出最高分辨率特征图，
    #即只需要将其他尺寸特征图的特征融合入最高分辨率特征图中
    #但在stage1中，self.multi_scale_output为True(多尺度输出)，所以range为2
    #i表示现在要把所有分支的特征（j）融合入第i分支的特征中

        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
    #对于j分支进行上采样或者下采样处理，使j分支的通道数以及shape等于i分支
            for j in range(num_branches):
    #j > i表示j通道多于i，但shape小于i，需要上采样
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        self.norm_layer(num_inchannels[i])))
    #j = i表示j与i为同一个分支，不需要做处理
                elif j == i:
                    fuse_layer.append(None)
    #剩余情况则是，j < i，表示j通道少于i，但shape大于i，需要下采样，利用一层或者多层conv2d进行下采样

                else:
                    conv3x3s = []
    #这个for k就是实现多层conv2d，而且只有最后一层加激活函数relu
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
    #在stage1中self.num_branches为2，所以不符合if条件
    #如果只有1个分支，就直接将单个分支特征图作为输入进入self.branches里设定的layers

        if self.num_branches == 1:
            return [self.branches[0](x[0])]
    #如果有多个分支，self.branches会是一个有两个元素（这里的元素是预设的layers）的列表
    #把对应的x[i]输入self.branches[i]即可
    #self.branches = self._make_branches(2, BASICBLOCK, [4,4], [32,64])


        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        
    #fuse_layers = self._make_fuse_layers()
    #现在已知self.fuse_layers里面有num_branches（上面的i）个元素fuse_layer
    #接下来就把不同的x分支输入到相应的self.fuse_layers元素中分别进行上采样和下采样
    #然后进行融合（相加实现融合）

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear',
                        align_corners=False
                        )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self,
                 cfg,
                 norm_layer=None):
        super(HighResolutionNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        # self.ASPP_1 = ASPP(64,64)
        # self.ASPP_2 = ASPP(18,18)
        # self.ASPP_3 = ASPP(36,36)
        # self.ASPP_4 = ASPP(72,72)
        self.ASPP_5 = ASPP(144,144)

        # stem network
        # stem net stem net,进行一系列的卷积操作,获得最初始的特征图N11
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False) #输入为3通道，输出为64通道
        self.bn1 = self.norm_layer(64)  #归一化层
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = self.norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

        # stage 1 layer 1
        self.stage1_cfg = cfg['STAGE1']
        """
        HRNET_18.STAGE1 = CN()
        HRNET_18.STAGE1.NUM_MODULES = 1        模型
        HRNET_18.STAGE1.NUM_BRANCHES = 1       分支为1
        HRNET_18.STAGE1.NUM_BLOCKS = [4]       4个块
        HRNET_18.STAGE1.NUM_CHANNELS = [64]    通道数为64
        HRNET_18.STAGE1.BLOCK = 'BOTTLENECK'   块为BOTTLENECK
        HRNET_18.STAGE1.FUSE_METHOD = 'SUM'    融合方式为 求和

        """
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0] # NUM_CHANNELS = [64]
        block = blocks_dict[self.stage1_cfg['BLOCK']]#BOTTLENECK
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0] #NUM_BLOCKS = 4
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        # stage 2
        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        # stage 3
        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        # stage 4
        self.stage4_cfg = cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

    def _make_transition_layer(  #这个完成创建分支和缩减图像shape     输入为先前的分支个数，和理想多少个数     #即增加一个尺度
            self, num_channels_pre_layer, num_channels_cur_layer):    
        #计算现在和以后有多少分支
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        #stage1的时候，num_branches_cur为2，所以有两个循环，i=0、1
        for i in range(num_branches_cur):
        #由于branches_cur有两个分支，branches_pre只有一个分支，
        #所以我们可以直接利用branches_pre已有分支作为branches_cur的其中一个分支
        #这个操作是hrnet的一个创新操作：在缩减特征图shape提取特征的同时，始终保留高分辨率特征图

            if i < num_branches_pre:
            #如果branches_cur通道数=branches_pre通道数，那么这个分支直接就可以用，不用做任何变化
            #如果branches_cur通道数！=branches_pre通道数，那么就要用一个cnn网络改变通道数
            #注意这个cnn是不会改变特征图的shape
            #在stage1中，pre通道数是256，cur通道数为32，所以要添加这一层cnn改变通道数
            #所以transition_layers第一层为
            #conv2d(256,32,3,1,1)
            #batchnorm2d(32)
            #rel

                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], #256    
                                  num_channels_cur_layer[i], #32   这里的shape  不会变
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        self.norm_layer(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
                    #由于branches_cur有两个分支，branches_pre只有一个分支
                    #所以我们必须要利用branches_pre里的分支无中生有一个新分支
                    #这就是常见的缩减图片shape，增加通道数提特征的操作

            else:
                conv3x3s = []
            #这里有一个for j作用：无论stage1的分支数为多少都能顺利构建模型
            #如果将stage1的分支设为3，那么需要生成2个新分支
            #第一个新分支需要由branches_pre最后一个分支缩减一次shape得到
            #但第二个新分支需要由branches_pre最后一个分支缩减两次shape得到，所以要做两次cnn，在第二次cnn才改变通道数
            #如果stage1分支设为4也是同样的道理
            #不过我们这里还是只考虑stage1分支为2的情况

                for j in range(i+1-num_branches_pre):
                    #利用branches_pre中shape最小，通道数最多的一个分支（即最后一个分支）来形成新分支
                    inchannels = num_channels_pre_layer[-1]
                    #outchannels为64
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),  #这里缩减1次shape
                        self.norm_layer(outchannels),
                        nn.ReLU(inplace=True)))
            #所以transition_layers第二层为:
            #nn.Conv2d(256, 64, 3, 2, 1, bias=False),
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True)

                transition_layers.append(nn.Sequential(*conv3x3s)) #这里是连接缩减1次shape之后的特征图

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):   #这里是layer层
        downsample = None
    #我们来看一下下面的if部分
    #在layer1中，block传入的是Bottlenect类，block.expansion是block类里的一个变量，定义为4
    #layer1的stride为1，planes为64，而self.inplane表示当前特征图通道数，经过初步提特征处理后的特征图通道数为是64，block.expanson=4，达成条件
    #那么downsample = nn.Sequential(
    #        nn.Conv2d(64, 64*4,kernel_size=1, stride=1, bias=False),
    #        nn.BatchNorm2d(64*4, momentum=BN_MOMENTUM),
    #    )
    #这里的downsample会在后面的bottleneck里面用到，用于下面block中调整输入x的通道数，实现残差结构相加


        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        #所以layers里第一层是：bottleneck(64, 64, 1, downsample)	(w,h,64)-->(w,h,256)	详细的分析在下面哦
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=self.norm_layer))
        #经过第一层后，当前特征图通道数为256
        inplanes = planes * block.expansion
        #这里的block为4，即for i in range(1,4)
        #所以这里for循环实现了3层bottleneck，目的应该是为了加深层数
        #bottleneck(256, 64, 1)  这里就没有传downsample了哦，因为残差结构相加不需要升维或者降维
        #bottleneck(256, 64, 1)
        #bottleneck(256, 64, 1)

        for i in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']       #1
        num_branches = layer_config['NUM_BRANCHES']     #2
        num_blocks = layer_config['NUM_BLOCKS']         #[4,4]
        num_channels = layer_config['NUM_CHANNELS']     #[32,64]
        block = blocks_dict[layer_config['BLOCK']]      #BASICBLOCK
        fuse_method = layer_config['FUSE_METHOD']       #SUM

        modules = []
    #num_modules表示一个融合块中要进行几次融合，前几次融合是将其他分支的特征融合到最高分辨率的特征图上，只输出最高分辨率特征图（multi_scale_output = False）
    #只有最后一次的融合是将所有分支的特征融合到每个特征图上，输出所有尺寸特征图（multi_scale_output=True）


        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
    #modules第一层是 HighResolutionModule(2,BASICBLOCK,[4,4],[32,64],[32,64],SUM,reset_multi_scale_output=True)
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output,
                                     norm_layer=self.norm_layer)
            )
             #获取现在各个分支有多少通道
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels


    def forward(self, x):
        features = []
        mixed_featurs = []
        list18 = []
        list36 = []
        list72 = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        features.append(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        list18.append(x)

        # 在forward函数中，初步提特征后下一行是：
        x = self.layer1(x)
        
        x_list = []

         #我们先看这个循环条件，在配置文件中self.stage2_cfg['NUM_BRANCHES']为2（其实总结构图上不也是画着两个分支嘛，分支也可以理解为有多少份不同尺寸的特征图）
         #所以这里有两个循环，i=0或1
         #在init中，有几行代码与self.transition1[i]有关，我们先搞清楚self.transition1[i]里到底是啥
        for i in range(self.stage2_cfg['NUM_BRANCHES']):   #2
            if self.transition1[i] is not None:
#   transition1[i]：
# 
#       extra['STAGE2']为
#       STAGE2:
#       NUM_MODULES: 1
#       NUM_BRANCHES: 2
#       BLOCK: BASIC
#       NUM_BLOCKS:
#       - 4
#       - 4
#       NUM_CHANNELS:
#       - 32
#       - 64
#       FUSE_METHOD: SUM
#                                                  
# self.stage2_cfg = extra['STAGE2']
#     #num_channels此时为[32,64]，
#     num_channels = self.stage2_cfg['NUM_CHANNELS']
#     #block为basic，传入的是一个类BasicBlock，因为代码中定义了一个blocks_dict = {'BASIC': BasicBlock,'BOTTLENECK': Bottleneck}
#     block = blocks_dict[self.stage2_cfg['BLOCK']]
#     #num_channels =[32*1,64*1]，这里num_channels的意义是stage2中，各个分支的通道数，这里乘1是因为basicblock里面expansion是1，即残差结构不会扩展通道数
#     num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
#     #这里有引入一个新的函数self._make_transition_layer
#     self.transition1 = self._make_transition_layer([256], num_channels)


                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        #上面的代码就是增加分支了
        #现在x_list里面有2个分支
        #self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)，这里是用来做提取特征和特征融合的
        #这里num_channels和上面的一样，是[32,64]
 
        y_list = self.stage2(x_list) #输入的是几个分支，输出的是融合后的分支
        
#       - 64
        
        list18.append(y_list[0])
       
        list36.append(y_list[1])
        
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        list18.append(y_list[0])
        list36.append(y_list[1])
        list72.append(y_list[2])
        
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
                    # here generate new scale features (downsample) 
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        list18.append(x[0])
        list36.append(x[1])
        list72.append(x[2])
        x[3] = self.ASPP_5(x[3])
        mixed_features = [list18] + [list36] + [list72] + [x[3]]
        #visual_feature(list18) 

      
        return features + mixed_features
        


def _hrnet(arch, pretrained, progress, **kwargs):
    from .hrnet_config import MODEL_CONFIGS
    model = HighResolutionNet(MODEL_CONFIGS[arch], **kwargs)
    if pretrained:
        if arch == 'hrnet64':
            arch = 'hrnet32_imagenet'
            model_url = model_urls[arch]
            loaded_state_dict = load_state_dict_from_url(model_url,
                                                  progress=progress)
            #add weights demention to adopt input change
            exp_layers = ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'conv2.weight', 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var']
            lista = ['transition1.0.0.weight', 'transition1.1.0.0.weight', 'transition2.2.0.0.weight', 'transition3.3.0.0.weight']
            for k,v in loaded_state_dict.items() :
                if k not in exp_layers:
                    if ('layer' not in k) and 'conv' in k or k in lista and len(v.size()) > 1:
                        if k in ['transition1.0.0.weight' , 'transition1.1.0.0.weight']:
                            loaded_state_dict[k] = torch.cat([loaded_state_dict[k]] * 2,0)
                        else:
                            loaded_state_dict[k] = torch.cat([v] * 2, 1) / 2
                            loaded_state_dict[k] = torch.cat([loaded_state_dict[k]] * 2,0)

                    if 'fuse_layer' in k and 'weight' in k and len(v.size()) > 1:
                        loaded_state_dict[k] = torch.cat([v] * 2, 1) / 2
                        loaded_state_dict[k] = torch.cat([loaded_state_dict[k]] * 2,0)

                    if 'layer' not in k and len(v.size()) == 1:
                        v = v.unsqueeze(1)
                        v = torch.cat([v] * 2, 0) 
                        loaded_state_dict[k] = v.squeeze(1) 
                    if 'fuse_layer' in k and len(v.size()) == 1:
                        v = v.unsqueeze(1)
                        v = torch.cat([v] * 2, 0) 
                        loaded_state_dict[k] = v.squeeze(1) 
                    if len(loaded_state_dict[k].size()) == 2:
                        loaded_state_dict[k] = loaded_state_dict[k].squeeze(1)
                    # for multi-input 
                    #if k == 'conv1.weight':
                    #  loaded_state_dict[k] = torch.cat([v] * 2, 1) / 2
        else:
            arch = arch + '_imagenet'
            model_url = model_urls[arch]
            loaded_state_dict = load_state_dict_from_url(model_url,
                                                  progress=progress)
        #if k == 'conv1.weight':
        #    loaded_state_dict[k] = torch.cat([v] * 2, 1) / 2
                   
        model.load_state_dict({k: v for k,v in loaded_state_dict.items() if k in model.state_dict()},strict = False)
    return model


def hrnet18(pretrained=True, progress=True, **kwargs):
    r"""HRNet-18 model
    """
    return _hrnet('hrnet18', pretrained, progress,
                   **kwargs)


def hrnet32(pretrained=True, progress=True, **kwargs):
    r"""HRNet-32 model
    """
    return _hrnet('hrnet32', pretrained, progress,
                   **kwargs)


def hrnet48(pretrained=True, progress=True, **kwargs):
    r"""HRNet-48 model
    """
    return _hrnet('hrnet48', pretrained, progress,
                   **kwargs)

def hrnet64(pretrained=True, progress=True, **kwargs):
    r"""HRNet-64 model
    """
    return _hrnet('hrnet64', pretrained, progress,
                   **kwargs)
