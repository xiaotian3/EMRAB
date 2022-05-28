import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.utils.model_zoo as model_zoo
from torchsummary import summary


def make_model(args, parent=False):
    return MODEL(args)

class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_channels, out_channels, groups=1):
   
    return nn.Conv2d(
        in_channels, 
        out_channels, 
        kernel_size=1, 
        groups=groups,stride=1)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

#EMRAB
class EMRAB(nn.Module):
    
    def __init__(self, n_feats, kernel_size, wn,  act=nn.ReLU(True)):
        super(EMRAB, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5

        self.ca = ChannelAttention(n_feats)
        self.sa = SpatialAttention()
        self.compress = conv1x1(3*n_feats, n_feats, 1)  
        self.compress1 = conv1x1(4*n_feats, n_feats, 1) 

        self.conv_3_1 = wn(nn.Conv2d(n_feats, n_feats, kernel_size_1,padding=1))
        self.conv_3_2 = wn(nn.Conv2d(n_feats, n_feats, kernel_size_1,padding=1))
        self.conv_3_3 = wn(nn.Conv2d(n_feats, n_feats, kernel_size_1,padding=1))

        self.conv_5_1 = wn(nn.Conv2d(n_feats, n_feats, kernel_size_2,padding=2))
        self.conv_5_2 = wn(nn.Conv2d(n_feats, n_feats, kernel_size_2,padding=2))
        self.conv_5_3 = wn(nn.Conv2d(n_feats, n_feats, kernel_size_2,padding=2))

        self.confusion = nn.Conv2d(n_feats *4, n_feats, 1, padding=0, stride=1) #2
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):        
        input_1 = x
        #input_2 = x
        
        output_3_1 = self.relu(self.conv_3_1(input_1))
        output_5_1 = self.relu(self.conv_5_1(input_1))
        input_2 = torch.cat([output_3_1, output_5_1,x], 1) #3*n_feats
        input_2 = self.compress(input_2) #n_feats  3n->1n
        
        output_3_2 = self.relu(self.conv_3_2(input_2)) #n_feats  
        output_5_2 = self.relu(self.conv_5_2(input_2)) #n_feats
        #input_3 = torch.cat([output_3_2, output_5_2], 1) #2*n_feats v1
        input_3 = torch.cat([output_3_2, output_5_2,x,input_2], 1) #4*n_feats
        input_3 = self.compress1(input_3) #n_feats 4n->1n
        
        output_3_3 = self.relu(self.conv_3_3(input_3)) #n_feats  
        output_5_3 = self.relu(self.conv_5_3(input_3)) #n_feats
        #input_4 = torch.cat([output_3_3, output_5_3], 1) #2*n_feats #v1
        input_4 = torch.cat([output_3_3, output_5_3,input_2,input_3], 1) #2*n_feats #v1
        #input_4 = self.compress(input_4) #n_feats
        
        output = self.confusion(input_4)
        #print( list(  output.size()  ) )
        output = self.ca(output) * output
        output = self.sa(output) * output
        output += x
        
        return output

class PA(nn.Module):

    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


class LFB(nn.Module):
    def __init__(
            self, n_feats, kernel_size, wn, act=nn.ReLU(True)):
        super(LFB, self).__init__()
#        self.b0 = EMRAB(n_feats, kernel_size, wn=wn, act=act)
#        self.b1 = EMRAB(n_feats, kernel_size, wn=wn, act=act)
#        self.b2 = EMRAB(n_feats, kernel_size, wn=wn, act=act)
#        self.b3 = EMRAB(n_feats, kernel_size, wn=wn, act=act)
#       self.reduction = wn(nn.Conv2d(n_feats * 4, n_feats, 3, padding=3 // 2))
        self.n = 4
        self.lfl=nn.ModuleList([EMRAB(n_feats, kernel_size, wn=wn, act=act)
            for i in range(self.n)])
            
        self.reduction = wn(nn.Conv2d(n_feats*self.n, n_feats, 3, padding=3//2))
        
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)

        # self.PA = PA(32)

    def forward(self, x):
#        x0 = self.b0(x)
#        # x0 = self.PA(x0)
#        x1 = self.b1(x0)
#        # x1 = self.PA(x1)
#        x2 = self.b2(x1)
#        # x2 = self.PA(x2)
#        x3 = self.b3(x2)
#        # x3 = self.PA(x3)
#        res = self.reduction(torch.cat([x0, x1, x2, x3], dim=1))
        
        
        #return self.res_scale(x3) + self.x_scale(x)
        s = x
        out=[]
        for i in range(self.n):
            x = self.lfl[i](x)
            out.append(x)
        res = self.reduction(torch.cat(out,dim=1))
        return self.res_scale(res) + self.x_scale(x)
    ##


class AWMS(nn.Module):
    def __init__(
            self, args, scale, n_feats, kernel_size, wn):
        super(AWMS, self).__init__()
        out_feats = scale * scale * args.n_colors
        self.tail_k3 = wn(nn.Conv2d(n_feats, out_feats, 3, padding=3 // 2, dilation=1))
        self.tail_k5 = wn(nn.Conv2d(n_feats, out_feats, 5, padding=5 // 2, dilation=1))
        self.tail_k7 = wn(nn.Conv2d(n_feats, out_feats, 7, padding=7 // 2, dilation=1))
        self.tail_k9 = wn(nn.Conv2d(n_feats, out_feats, 9, padding=9 // 2, dilation=1))
        self.pixelshuffle = nn.PixelShuffle(scale)
        self.scale_k3 = Scale(0.25)
        self.scale_k5 = Scale(0.25)
        self.scale_k7 = Scale(0.25)
        self.scale_k9 = Scale(0.25)

    def forward(self, x):
        x0 = self.pixelshuffle(self.scale_k3(self.tail_k3(x)))
        x1 = self.pixelshuffle(self.scale_k5(self.tail_k5(x)))
        x2 = self.pixelshuffle(self.scale_k7(self.tail_k7(x)))
        x3 = self.pixelshuffle(self.scale_k9(self.tail_k9(x)))

        return x0 + x1 + x2 + x3

class MODEL(nn.Module):
    def __init__(self, args):
        super(MODEL, self).__init__()
        # hyper-params
        self.args = args
        scale = args.scale[0]
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)
        # wn = lambda x: x
        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(
            [0.4488, 0.4371, 0.4040])).view([1, 3, 1, 1])

        # def _make_layer(self, block, planes, blocks, stride=1):
        #    downsample = None
        #    if stride != 1 or self.inplanes != planes * block.expansion:
        #        downsample = nn.Sequential(
        #            nn.Conv2d(self.inplanes, planes * block.expansion,
        #                      kernel_size=1, stride=stride, bias=False),
        #            nn.BatchNorm2d(planes * block.expansion),
        #        )
        # self.msa = attention.PyramidAttention(channel=256, reduction=8,res_scale=args.res_scale);

        # define head module
        # head = HEAD(args, n_feats, kernel_size, wn)
        head = []
        head.append(
            wn(nn.Conv2d(args.n_colors, n_feats, 3, padding=3 // 2)))

        # define body module
        body = []
        # body.append(self.msa)
        for i in range(n_resblocks):
            body.append(
                LFB(n_feats, kernel_size, wn=wn, act=act))

        # define tail module
        out_feats = scale * scale * args.n_colors
        tail = AWMS(args, scale, n_feats, kernel_size, wn)

        skip = []
        # for i in range(2):
        # skip.append(
        # wn(nn.Conv2d(args.n_colors, out_feats, 3, padding=3//2)))

        skip.append(
            wn(nn.Conv2d(args.n_colors, out_feats, 3, padding=3 // 2)))
        skip.append(nn.PixelShuffle(scale))

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = tail
        self.skip = nn.Sequential(*skip)

    def forward(self, x):
        x = (x - self.rgb_mean.cuda() * 255) / 127.5
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
        x = x * 127.5 + self.rgb_mean.cuda() * 255
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0 or name.find('skip') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

