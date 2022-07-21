import torch
from torch import nn
import torch.nn.functional as F
import math
from backbone.repvgg import get_RepVGG_func_by_name
import utils
import math


def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup), nn.ReLU(inplace=True))

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio,
                      inp * expand_ratio,
                      3,
                      stride,
                      1,
                      groups=inp * expand_ratio,
                      bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
class PFLDInference(nn.Module):
    def __init__(self,
                 backbone_name, backbone_file, deploy,
                 bins=(1, 2, 3, 6),
                 droBatchNorm=nn.BatchNorm2d,
                 pretrained=True):
        super(PFLDInference, self).__init__()

        repvgg_fn = get_RepVGG_func_by_name(backbone_name)
        backbone = repvgg_fn(deploy)
        if pretrained:
            checkpoint = torch.load(backbone_file)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k,
                    v in checkpoint.items()}  # strip the names
            backbone.load_state_dict(ckpt)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = backbone.stage0, backbone.stage1, backbone.stage2, backbone.stage3, backbone.stage4
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        last_channel = 0
        for n, m in self.layer4.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                last_channel = m.out_channels

        fea_dim = last_channel

        self.linear_reg = nn.Linear(fea_dim, 6)

    def forward(self, x):

        x = self.layer0(x)
        out1 = self.layer1(x)
        
        x = self.layer2(out1)
        
        x = self.layer3(x)
        x = self.layer4(x)
        x= self.gap(x)
        x = torch.flatten(x, 1)
        x = self.linear_reg(x)
        return out1, utils.compute_rotation_matrix_from_ortho6d(x)
    
class AuxiliaryNet(nn.Module):
    def __init__(self):
        super(AuxiliaryNet, self).__init__()
        self.conv1 = conv_bn(128, 128, 3, 1)
        self.conv2 = conv_bn(128, 128, 3, 2)
        self.conv3 = conv_bn(128, 256, 3, 1)
        self.conv4 = conv_bn(256, 128, 3, 2)
        self.max_pool1 = nn.MaxPool2d(3)
        self.conv5 = conv_bn(128, 32, 3, 2)
        self.avg_pool1 = nn.AvgPool2d(4)
        self.avg_pool2 = nn.AvgPool2d(2)
        self.fc = nn.Linear(672, 196)
        self.relu = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(32, 128, 1, 1, 0)


    def forward(self, x):
        #print(x.shape) #torch.Size([1, 24, 56, 56])
        x = self.conv1(x)
        #print(x.shape)#torch.Size([1, 128, 56, 56])
        x = self.conv2(x)
        #print(x.shape)#torch.Size([1, 128, 28, 28])
        x = self.conv3(x)
        #print(x.shape)#torch.Size([1, 256, 28, 28])
        x = self.conv4(x)
        #print(x.shape)#torch.Size([1, 128, 14, 14])
        x = self.max_pool1(x)
        
        #print(x.shape)#torch.Size([1, 128, 4, 4])
        x1 = self.avg_pool1(x)
        #print(x1.shape)#torch.Size([1, 128, 1, 1])
        x1 = x1.view(x1.size(0), -1)
        

        x = self.conv5(x)
        #print(x.shape)#torch.Size([1, 32, 2, 2])
        x2 = self.avg_pool2(x)
        #print(x2.shape)#torch.Size([1, 32, 1, 1])
        x2 = x2.view(x2.size(0), -1)
        
        #print(x.shape)#torch.Size([1, 32, 2, 2])
        x3 = self.relu(self.conv6(x))
        #print(x3.shape)#torch.Size([1, 128, 2, 2])
        x3 = x3.view(x3.size(0), -1)
        
        
        multi_scale = torch.cat([x1, x2, x3], 1)
        
        landmarks = self.fc(multi_scale)
        
        return landmarks
