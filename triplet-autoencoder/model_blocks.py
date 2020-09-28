import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class UnetDoubleConv(nn.Module):
    """(conv -> bn -> relu)x2"""
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels * 2

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UnetDown(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            UnetDoubleConv(in_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UnetUp(nn.Module):
    """resize_mode: None, nearest, linear, bilinear, bicubic"""
    def __init__(self, in_channels, resize_mode=None):
        super().__init__()

        if resize_mode is None:
            self.up = nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2)
        elif resize_mode == 'nearest':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=resize_mode),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2,mode=resize_mode,align_corners=True),
                nn.Conv2d(in_channels,in_channels//2,kernel_size=3,padding=1),
                nn.BatchNorm2d(in_channels//2),
                nn.ReLU(inplace=True)
            )
        self.conv = UnetDoubleConv(in_channels,in_channels//2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_H = x2.size()[2] - x1.size()[2]
        diff_W = x2.size()[3] - x1.size()[3]

        if diff_H != 0 or diff_W != 0:
            x1 = F.pad(x1,[diff_H//2,diff_H-diff_H//2,
                           diff_W//2,diff_W-diff_W//2])

        "NCHW"
        x = torch.cat([x2,x1],dim=1)
        return self.conv(x)

class ConvUp(nn.Module):
    """resize_mode: None, nearest, linear, bilinear, bicubic"""
    def __init__(self, in_channels, resize_mode=None):
        super().__init__()

        if resize_mode is None:
            self.up = nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2)
        elif resize_mode == 'nearest':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=resize_mode),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2,mode=resize_mode,align_corners=True),
                nn.Conv2d(in_channels,in_channels//2,kernel_size=3,padding=1),
                nn.BatchNorm2d(in_channels//2),
                nn.ReLU(inplace=True)
            )
        self.conv = UnetDoubleConv(in_channels//2,in_channels//2)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class UnetOutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DenseSqential(nn.Module):
    def __init__(self,input_channels,widths):
        super().__init__()
        if isinstance(widths,int):
            self.net = nn.Sequential(
                nn.Linear(in_features=input_channels,out_features=widths),
                # nn.Tanh()
            )
        else:
            fulls = []
            for i,n in enumerate(widths):
                if i==0:
                    fulls.append(nn.Linear(in_features=input_channels,out_features=n))
                    # fulls.append(nn.Tanh())
                else:
                    fulls.append(nn.Linear(in_features=widths[i-1],out_features=n))
                    # fulls.append(nn.Tanh())
            self.net = nn.Sequential(*fulls)

    def forward(self, x):
        return self.net(x)

class UnetEncoder(nn.Module):
    def __init__(self,image_size,input_channels,model_depth,model_width,code_width):
        super().__init__()

        self.image_size = image_size
        self.input_channels = input_channels
        self.model_depth = model_depth
        self.model_width = model_width
        self.code_width = code_width

        self.inc = UnetDoubleConv(input_channels, model_width)
        self.down = []

        for i in range(model_depth):
            self.down.append(UnetDown(model_width * (2**i)))
        self.down = nn.ModuleList(self.down)

        if code_width is not None:
            self.full = DenseSqential((image_size//(2**model_depth))**2*model_width*2**model_depth, code_width)
            if isinstance(code_width,int):
                self.n_feature = code_width
            else:
                self.n_feature = code_width[-1]
        else:
            self.full = torch.nn.Identity()
            self.n_feature = (image_size//(2**model_depth))**2*model_width*2**model_depth

    def forward(self,x):
        x = self.inc(x)
        for m in self.down:
            x = m(x)
        return self.full(x.reshape(x.shape[0],-1))

class SiameseLoss(nn.Module):
    def __init__(self,m):
        super().__init__()
        self.m = m
    def forward(self,x):
        return torch.sum(torch.clamp(self.m+x,0))

class SiamInvarLoss(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, logits):
        x = logits[0] - logits[1]
        return torch.mean(torch.clamp(self.m + x, 0))

class EncodeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, raw_input):
        L1 = F.mse_loss(logits[2],raw_input[0],reduction='mean')
        L2 = F.mse_loss(logits[3],raw_input[2],reduction='mean')
        return L1 + L2

class ConvBnReLu(nn.Sequential):
    def __init__(self,in_channel,out_channel,kernal_size=3,stride=1,padding=1,dilation=1,momentum=0.999,relu=True):
        super().__init__()
        self.add_module(
            'conv',
            nn.Conv2d(in_channel,out_channel,kernal_size,stride,padding,dilation)
        )
        self.add_module(
            'bn',
            nn.BatchNorm2d(out_channel,momentum=momentum)
        )
        if relu:
            self.add_module(
                'relu',
                nn.ReLU(inplace=True)
            )

class ResLayer(nn.Module):
    def __init__(self,in_channel,out_channel,stride,dilation,short_cut_projection):
        super().__init__()
        mid_channel = out_channel//4
        self.reduce = ConvBnReLu(in_channel,mid_channel,1,stride,0,dilation=1)
        self.conv3x3 = ConvBnReLu(mid_channel,mid_channel,3,1,dilation,dilation) #same padding
        self.increase = ConvBnReLu(mid_channel,out_channel,1,1,0,dilation=1,relu=False)
        if short_cut_projection:
            self.shortcut = ConvBnReLu(in_channel,out_channel,1,stride,0,dilation=1,relu=False)
        else:
            self.shortcut = lambda x:x

    def forward(self, x):
        x1 = self.reduce(x)
        x1 = self.conv3x3(x1)
        x1 = self.increase(x1)
        return F.relu(x1+self.shortcut(x),inplace=True)

class ResBlock(nn.Sequential):
    def __init__(self,n_layer,in_channel,out_channel,stride,dilation,grids=None):
        super().__init__()
        if grids is None:
            grids = [1 for _ in range(n_layer)]
        else:
            assert n_layer == len(grids)

        for i in range(n_layer):
            self.add_module(
                f"block_{i+1}",
                ResLayer(
                    in_channel = in_channel if i==0 else out_channel,
                    out_channel = out_channel,
                    stride = stride if i==0 else 1,
                    dilation = dilation * grids[i],
                    short_cut_projection= i==0
                )
            )

class ResStem(nn.Sequential):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.add_module("conv1",ConvBnReLu(in_channel,out_channel,7,2,3,1))
        self.add_module("pool",nn.MaxPool2d(3,2,1))

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0),-1)

class ImagePool(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = ConvBnReLu(in_channel,out_channel,kernal_size=1,stride=1,padding=0,dilation=1)

    def forward(self, x):
        _,_,H,W = x.shape
        x = self.pool(x)
        x = self.conv(x)
        x = F.interpolate(x,size=(H,W),mode='bilinear',align_corners=False)
        return x

class ASPPModule(nn.Module):
    def __init__(self,in_channel,out_channel,rates):
        super().__init__()
        self.stages = []
        self.stages.append(
            ConvBnReLu(in_channel,out_channel,kernal_size=1,stride=1,padding=0,dilation=1)
        )
        for r in rates:
            self.stages.append(
                ConvBnReLu(in_channel,out_channel,kernal_size=3,stride=1,
                           padding=r,dilation=r)
            )
        self.stages.append(ImagePool(in_channel,out_channel))
        self.stages = nn.ModuleList(self.stages)

    def forward(self,x):
        res = []
        for s in self.stages:
            res.append(s(x))
        return torch.cat(res,dim=1)

class BinaryDiceLoss(nn.Module):
    def __init__(self,smooth=1):
        super().__init__()
        self.smooth = smooth
    def forward(self, predict,target):
        N = predict.shape[0]
        predict = predict.contiguous().view(N,-1)
        target = target.contiguous().view(N,-1)
        num = torch.sum(torch.mul(predict,target),dim=1) + self.smooth
        den = torch.sum(predict.pow(2) + target.pow(2), dim=1) + self.smooth
        return (1 - num / den).mean()

class DiceLoss(nn.Module):
    def __init__(self,smooth = 1):
        super().__init__()
        self.dl = BinaryDiceLoss(smooth = smooth)

    def forward(self, logits, target, **kwargs):
        "logits [N,C,H,W], target [N,1,H,W]"
        predict = F.softmax(logits,dim=1)
        n_class = logits.shape[1]
        target = DiceLoss.one_hot(target,n_class)
        if torch.cuda.is_available():
            target = target.cuda()
        if 'weight' not in kwargs.keys():
            weight = [1.0 for _ in range(n_class)]
        else:
            weight = kwargs['weight']
        total_loss = 0
        for i in range(n_class):
            total_loss += weight[i] * self.dl(predict[:,i],target[:,i])
        return total_loss

    @staticmethod
    def one_hot(tag,n_class):
        r = torch.zeros(tuple([tag.shape[0],n_class,tag.shape[2],tag.shape[3]]))
        return r.scatter_(1,tag.cpu(),1)

class SelfSegLoss(nn.Module):
    def __init__(self,reduce_mode='minmax'):
        super().__init__()
        self.w = SelfSegLoss.genW()
        self.reduce_mode = reduce_mode

    def forward(self, logits, target):
        D = SelfSegLoss.LayerConv2d(logits,self.w)
        Z = SelfSegLoss.LayerConv2d(target.double(),self.w)
        b = Z>0.5
        dout = torch.mul(D,b).contiguous().view(-1)
        din = torch.mul(D,torch.logical_not(b)).contiguous().view(-1)
        b = b.contiguous().view(-1)
        dout = dout[b]
        din = din[torch.logical_not(b)]
        if self.reduce_mode == 'minmax':
            return torch.clamp(10.0 + din.max() - dout.min(), 0)
        elif self.reduce_mode == 'mean':
            return torch.clamp(10.0 + din.mean() - dout.mean(), 0)
        elif self.reduce_mode == 'expmean':
            return torch.clamp(10.0 + (din+1.0).pow(1.5).mean() - (1.0/(dout+0.1)).mean(), 0)


    @staticmethod
    def genW():
        w = []
        for i in range(9):
            if i != 4:
                w.append(utils.gen9DirWeight(i).unsqueeze(dim=0))
        w = torch.cat(w,dim=0) #[8,3,3]
        w = w.unsqueeze(dim=1).double()#[8,1,3,3]
        return w.cuda() if torch.cuda.is_available() else w

    @staticmethod
    def LayerConv2d(x,w,padding=1,pow=2):
        C = x.shape[1]
        NF = w.shape[0]
        r = []
        for f in range(NF):
            tmp = torch.zeros((x.shape[0],1,x.shape[2],x.shape[3]),dtype=x.dtype)
            if torch.cuda.is_available():
                tmp = tmp.cuda()
            w_in = w[f,:,:,:].unsqueeze(dim=0)
            for i in range(C):
                x_in = x[:,i,:,:].unsqueeze(dim=1)
                tmp += F.conv2d(x_in,w_in,padding=padding).pow(pow)
            r.append(tmp)
        return torch.cat(r,dim=1)

class StairCELoss(nn.Module):
    def __init__(self,method,align,weight=None):
        super().__init__()
        if weight is None:
            weight = [1.0, 0.75, 0.5, 0.25]
        self.weight = weight
        self.method = method
        self.align = align

    def forward(self, xs, label):
        assert len(xs) == 4
        L = 0
        for i in range(4):
            if self.method != 'nearest':
                L += self.weight[i] * F.cross_entropy(xs[i],
                                                      F.interpolate(label.unsqueeze(1).float(),
                                                                    scale_factor=1.0/2.0**i,mode=self.method,
                                                                    align_corners=self.align).squeeze().long())
            else:
                L += self.weight[i] * F.cross_entropy(xs[i],
                                                      F.interpolate(label.unsqueeze(1).float(),
                                                                    scale_factor=1.0 / 2.0 ** i, mode='nearest').squeeze().long())

        return L

class Mean_Abs_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xs, label):
        return torch.sum(torch.abs(xs-label))


