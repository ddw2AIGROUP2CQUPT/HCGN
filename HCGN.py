# @Time    : 2024/06/01 21:09
# @Author  : Yang Xionghui
# @File    : HCGN.py
# @Software: PyCharm

import torch
import torch.nn as nn
from gcn_lib import DenseDilatedKnnGraph
from timm.models.layers import DropPath
import argparse

from torch_geometric.nn import GATConv, LayerNorm
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

from convnext_fcn import convnext_tiny
from vgg_fcn import vgg11, vgg13, vgg16, vgg19
from resnet_fcn import resnet18, resnet34, resnet50, resnet101, resnet152

class FFN(nn.Module):

    def __init__(self, in_channels, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels) # depthwise conv
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channels, 4 * in_channels) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * in_channels, in_channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((in_channels)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class attention(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.gate_nn = nn.Linear(hidden_channels, 1)
        self.nn = nn.ReLU()
    def forward(self, x, batch):
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.int64)
        size = batch[-1].item()+1
        gate = self.gate_nn(x)
        x = self.nn(x) if self.nn is not None else x
        attention = softmax(gate, batch, num_nodes=size)
        out = scatter_add(attention*x, batch, dim=0, dim_size=size)

        return out, attention

class GATLayer_mutihead(nn.Module):
    def __init__(self, in_channels, args):
        super(GATLayer_mutihead, self).__init__()
        self.in_channels = in_channels

        self.edge_index = DenseDilatedKnnGraph(k=args.k, dilation=1, stochastic=args.use_stochastic, epsilon=args.epsilon)

        self.conv1 = GATConv(in_channels=in_channels, out_channels=in_channels, heads=args.heads, dropout=0.0)
        self.norm1 = LayerNorm(in_channels=in_channels * args.heads)
        self.gelu1 = nn.GELU()

        self.conv_down = nn.Sequential(
                                        nn.Conv2d(in_channels * args.heads, in_channels, kernel_size=1), 
                                        nn.BatchNorm2d(in_channels), 
                                        nn.GELU()
                                        )
        

        self.ffn = FFN(in_channels=in_channels)

        # self.attention_mean = attention(in_channels * args.heads)

    def forward(self, x):
        tmp = x     #[b,c,w,h]

        b, c, w, h = x.shape
        adj = x.reshape(b, h*w, c)

        adj = self.edge_index(adj)

        x = x.reshape([b, c, -1]) # [b, c, h*w]
        x = x.transpose(1, 2)     # [b, h*w, c]
        x = x.reshape([-1, c])    # [b*h*w, c]

        # block 1
        x = self.conv1(x, adj)    # [b*h*w, c*8]
        x = self.norm1(x)
        x = self.gelu1(x)

        c = x.shape[-1]
        x = x.reshape([b, c, w, h]) #[b, c*8, w, h]


        x = self.conv_down(x)    #[b,c,w,h]

        x = self.ffn(x)

        x = tmp + x

        return x


class HCGN(nn.Module):

    def __init__(self, args, backbone, dims, pretrained=False, head_init_scale=1.):
        super().__init__()

        self.args = args

        self.backbone = backbone

        if pretrained == True:
            print('=======Model pretrained========')

        self.gnn_layer1 = GATLayer_mutihead(dims[0], args=args)
        self.gnn_layer2 = GATLayer_mutihead(dims[1], args=args)
        self.gnn_layer3 = GATLayer_mutihead(dims[2], args=args)
        # self.gnn_layer4 = GATLayer_mutihead(dims[3], args=args)

        self.norm = nn.LayerNorm(dims[3], eps=1e-6) # final norm layer

        self.head = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(dims[3], 4*dims[3]),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4*dims[3], args.num_classes),
            nn.Dropout(),
        )

        self.head[1].weight.data.mul_(head_init_scale)
        self.head[1].bias.data.mul_(head_init_scale)
        self.head[4].weight.data.mul_(head_init_scale)
        self.head[4].bias.data.mul_(head_init_scale)
            
    def forward(self, x):

        x = self.backbone[0](x) #[1, 96, 96, 96]
        x = self.gnn_layer1(x)

        x = self.backbone[1](x) #[1, 192, 48, 48]
        x = self.gnn_layer2(x)

        x = self.backbone[2](x) #[1, 384, 24, 24]
        x = self.gnn_layer3(x)
        
        x = self.backbone[3](x) #[1, 768, 12, 12]
        # x = self.gnn_layer4(x)

        x = self.norm(x.mean([-2, -1]))

        x = self.head(x)
        
        return x

def HCGN_VGG11(args):
    backbone = vgg11(pretrained=args.pretrained)
    dims = [256, 512, 512, 512]
    return HCGN(args=args, backbone=backbone, dims=dims, pretrained=args.pretrained)

def HCGN_VGG13(args):
    backbone = vgg13(pretrained=args.pretrained)
    dims = [256, 512, 512, 512]
    return HCGN(args=args, backbone=backbone, dims=dims, pretrained=args.pretrained)
    
def HCGN_VGG16(args):
    backbone = vgg16(pretrained=args.pretrained)
    dims = [256, 512, 512, 512]
    return HCGN(args=args, backbone=backbone, dims=dims, pretrained=args.pretrained)

def HCGN_VGG19(args):
    backbone = vgg19(pretrained=args.pretrained)
    dims = [256, 512, 512, 512]
    return HCGN(args=args, backbone=backbone, dims=dims, pretrained=args.pretrained)

def HCGN_RESNET18(args):
    backbone = resnet18(pretrained=args.pretrained)
    dims = [64, 128, 256, 512]
    return HCGN(args=args, backbone=backbone, dims=dims, pretrained=args.pretrained)

def HCGN_RESNET34(args):
    backbone = resnet34(pretrained=args.pretrained)
    dims = [64, 128, 256, 512]
    return HCGN(args=args, backbone=backbone, dims=dims, pretrained=args.pretrained)

def HCGN_RESNET50(args):
    backbone = resnet50(pretrained=args.pretrained)
    dims = [256, 512, 1024, 2048]
    return HCGN(args=args, backbone=backbone, dims=dims, pretrained=args.pretrained)

def HCGN_RESNET101(args):
    backbone = resnet101(pretrained=args.pretrained)
    dims = [256, 512, 1024, 2048]
    return HCGN(args=args, backbone=backbone, dims=dims, pretrained=args.pretrained)

def HCGN_RESNET152(args):
    backbone = resnet152(pretrained=args.pretrained)
    dims = [256, 512, 1024, 2048]
    return HCGN(args=args, backbone=backbone, dims=dims, pretrained=args.pretrained)

def HCGN_CONVNEXT_TINY(args):
    backbone = convnext_tiny(pretrained=args.pretrained,in_22k=True)
    dims = [96, 192, 384, 768]
    return HCGN(args=args, backbone=backbone, dims=dims, pretrained=args.pretrained)
    

