import torch
import torch.nn as nn
import torch.nn.functional as F
import model_blocks

class UNet(nn.Module):
    def __init__(self, input_channel, n_class, model_width, resize_method = None):
        super().__init__()
        self.input_channel = input_channel
        self.n_class = n_class
        self.model_width = model_width
        self.resize_method = resize_method

        self.inc = model_blocks.UnetDoubleConv(input_channel,model_width)
        self.down1 = model_blocks.UnetDown(model_width)
        self.down2 = model_blocks.UnetDown(model_width*2)
        self.down3 = model_blocks.UnetDown(model_width*4)
        self.down4 = model_blocks.UnetDown(model_width*8)

        self.up1 = model_blocks.UnetUp(model_width*16, resize_method)
        self.up2 = model_blocks.UnetUp(model_width*8, resize_method)
        self.up3 = model_blocks.UnetUp(model_width*4, resize_method)
        self.up4 = model_blocks.UnetUp(model_width*2, resize_method)

        self.out = model_blocks.UnetOutConv(model_width, n_class)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out(x)

class UNet2(nn.Module):
    def __init__(self, input_channel, n_class, model_width, resize_method = None):
        super().__init__()
        self.input_channel = input_channel
        self.n_class = n_class
        self.model_width = model_width
        self.resize_method = resize_method

        self.inc = model_blocks.UnetDoubleConv(input_channel,model_width)
        self.down1 = model_blocks.UnetDown(model_width)
        self.down2 = model_blocks.UnetDown(model_width*2)
        self.down3 = model_blocks.UnetDown(model_width*4)
        self.down4 = model_blocks.UnetDown(model_width*8)

        self.up1 = model_blocks.UnetUp(model_width*16, resize_method)
        self.up2 = model_blocks.UnetUp(model_width*8, resize_method)
        self.up3 = model_blocks.UnetUp(model_width*4, resize_method)
        self.up4 = model_blocks.UnetUp(model_width*2, resize_method)

        self.out = model_blocks.UnetOutConv(model_width, n_class)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return x, self.out(x)

class UNet3(nn.Module):
    def __init__(self, input_channel, n_class, model_width, resize_method = None):
        super().__init__()
        self.input_channel = input_channel
        self.n_class = n_class
        self.model_width = model_width
        self.resize_method = resize_method

        self.inc = model_blocks.UnetDoubleConv(input_channel,model_width)
        self.down1 = model_blocks.UnetDown(model_width)
        self.down2 = model_blocks.UnetDown(model_width*2)
        self.down3 = model_blocks.UnetDown(model_width*4)
        self.down4 = model_blocks.UnetDown(model_width*8)

        self.up1 = model_blocks.UnetUp(model_width*16, resize_method)
        self.up2 = model_blocks.UnetUp(model_width*8, resize_method)
        self.up3 = model_blocks.UnetUp(model_width*4, resize_method)
        self.up4 = model_blocks.UnetUp(model_width*2, resize_method)

        self.out1 = model_blocks.UnetOutConv(model_width, n_class)
        self.out2 = model_blocks.UnetOutConv(model_width*2, n_class)
        self.out3 = model_blocks.UnetOutConv(model_width*4, n_class)
        self.out4 = model_blocks.UnetOutConv(model_width*8, n_class)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out1(x), self.out2(x2), self.out3(x3), self.out4(x4)

class SiameseNet(nn.Module):
    def __init__(self,encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        x1 = self.encoder(x[0])
        x2 = self.encoder(x[1])
        x3 = self.encoder(x[2])
        dap = torch.pairwise_distance(x1,x2)
        dan = torch.pairwise_distance(x1,x3)
        return dap-dan

class RotInvarSiameseNet(nn.Module):
    def __init__(self,encoder,decoder=None,resize_method='nearest'):
        super().__init__()
        self.encoder = encoder
        if decoder is None:
            if isinstance(encoder.code_width, int):
                self.decoder = DecoderNet(encoder.image_size, encoder.n_feature, [], encoder.model_depth,
                                          resize_method, encoder.model_width, encoder.input_channels)
            elif encoder.code_width is None:
                self.decoder = DecoderNet(encoder.image_size, encoder.n_feature, None, encoder.model_depth,
                                          resize_method, encoder.model_width, encoder.input_channels)
            else:
                self.decoder = DecoderNet(encoder.image_size, encoder.n_feature, encoder.code_width[::-1][1:],
                                          encoder.model_depth,
                                          resize_method, encoder.model_width, encoder.input_channels)
        else:
            self.decoder = decoder


    def forward(self, x):
        x1 = self.encoder(x[0])
        x2 = self.encoder(x[1])
        x3 = self.encoder(x[2])
        dap = torch.pairwise_distance(x1, x2)
        dan = torch.pairwise_distance(x1, x3)
        return dap, dan, self.decoder(x1), self.decoder(x3)

class CellAutoEncoder(nn.Module):
    def __init__(self,encoder,resize_method='nearest'):
        super().__init__()
        self.encoder = encoder
        if isinstance(encoder.code_width,int):
            self.decoder = DecoderNet(encoder.image_size, encoder.n_feature, [],encoder.model_depth,
                                      resize_method, encoder.model_width, encoder.input_channels)
        elif encoder.code_width is None:
            self.decoder = DecoderNet(encoder.image_size, encoder.n_feature, None, encoder.model_depth,
                                      resize_method, encoder.model_width, encoder.input_channels)
        else:
            self.decoder = DecoderNet(encoder.image_size,encoder.n_feature, encoder.code_width[::-1][1:],encoder.model_depth,
                                      resize_method,encoder.model_width,encoder.input_channels)
        # self.out = nn.Sigmoid()


    def forward(self, x):
        x1 = self.encoder(x)
        x1 = self.decoder(x1)
        # return self.out(x1)
        return x1

class DecoderNet(nn.Module):
    def __init__(self,image_size,input_width,code_width,decoder_depth,resize_method,end_code_channel,end_channels):
        super().__init__()
        self.init_image_size = image_size // (2**decoder_depth)
        self.init_array_length = self.init_image_size * self.init_image_size * end_code_channel * 2 ** decoder_depth

        if code_width is not None:
            code_width = [x for x in code_width]
            code_width.append(self.init_array_length)
            self.dense = model_blocks.DenseSqential(input_width, code_width)
            self.in_channel = code_width[-1] // (self.init_image_size ** 2)
        else:
            self.dense = torch.nn.Identity()
            self.in_channel = input_width // (self.init_image_size ** 2)

        self.up = []
        in_channel = self.in_channel
        for i in range(decoder_depth):
            self.up.append(model_blocks.ConvUp(in_channels=in_channel,resize_mode=resize_method))
            in_channel = in_channel // 2
        self.out = model_blocks.UnetOutConv(in_channel, end_channels)
        self.up = nn.ModuleList(self.up)

    def forward(self, x):
        x = self.dense(x)
        x = torch.reshape(x,(x.shape[0],self.in_channel,self.init_image_size,self.init_image_size))
        for m in self.up:
            x = m(x)
        return self.out(x)

class Deeplabv1_like(nn.Module):
    def __init__(self,in_channel,n_class,n_blocks, width=64):
        super().__init__()
        ch = [width * 2 ** p for p in range(6)]
        self.C = model_blocks.ResStem(in_channel, ch[0])
        self.D1 = model_blocks.ResBlock(n_blocks[0], ch[0], ch[2], 1, 1)
        self.D2 = model_blocks.ResBlock(n_blocks[1], ch[2], ch[3], 2, 1)
        self.D3 = model_blocks.ResBlock(n_blocks[2], ch[3], ch[4], 1, 2)
        self.D4 = model_blocks.ResBlock(n_blocks[3], ch[4], ch[5], 1, 4)
        self.out = nn.Sequential(
            nn.Conv2d(ch[5], n_class, 1),
            nn.Upsample(scale_factor=8))

    def forward(self, x):
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        x = self.L4(x)
        x = self.L5(x)
        return self.out(x)

class ResUNet(nn.Module):
    def __init__(self,in_channel,n_class,n_blocks, width=64):
        super().__init__()
        ch = [width * 2 ** p for p in range(5)]

        self.C = model_blocks.UnetDoubleConv(in_channel,ch[0])
        self.L1 = model_blocks.ResBlock(n_blocks[0], ch[0], ch[1], 2, 1)
        self.L2 = model_blocks.ResBlock(n_blocks[1], ch[1], ch[2], 2, 1)
        self.L3 = model_blocks.ResBlock(n_blocks[2], ch[2], ch[3], 2, 1)
        self.L4 = model_blocks.ResBlock(n_blocks[3], ch[3], ch[4], 2, 1)

        self.U1 = model_blocks.UnetUp(ch[4], 'nearest')
        self.U2 = model_blocks.UnetUp(ch[3], 'nearest')
        self.U3 = model_blocks.UnetUp(ch[2], 'nearest')
        self.U4 = model_blocks.UnetUp(ch[1], 'nearest')

        self.out = model_blocks.UnetOutConv(width, n_class)

    def forward(self, x):
        x1 = self.C(x)
        x2 = self.L1(x1)
        x3 = self.L2(x2)
        x4 = self.L3(x3)
        x5 = self.L4(x4)

        x = self.U1(x5, x4)
        x = self.U2(x, x3)
        x = self.U3(x, x2)
        x = self.U4(x, x1)

        return self.out(x)

class Deeplabv3plus(nn.Module):
    def __init__(self,in_channel,n_class,n_blocks,atrous_rates,width,multigrids,output_stride=8):
        super().__init__()
        if output_stride == 8:
            s = [1,2,1,1]
            d = [1,1,2,4]
        elif output_stride == 16:
            s = [1,2,2,1]
            d = [1,1,1,1]
        else:
            raise ValueError(f'output_stride only accept 8 or 16, given {output_stride}')

        ch = [width * 2 ** p for p in range(6)]
        self.C = model_blocks.ResStem(in_channel,ch[0])
        self.L1 = model_blocks.ResBlock(
            n_blocks[0],ch[0],ch[2],stride=s[0],dilation=d[0]
        )
        self.L2 = model_blocks.ResBlock(
            n_blocks[1], ch[2], ch[3], stride=s[1], dilation=d[1]
        )
        self.L3 = model_blocks.ResBlock(
            n_blocks[2], ch[3], ch[4], stride=s[2], dilation=d[2]
        )
        self.L4 = model_blocks.ResBlock(
            n_blocks[3], ch[4], ch[5], stride=s[3], dilation=d[3], grids=multigrids
        )
        self.aspp = model_blocks.ASPPModule(
            ch[5],256,atrous_rates
        )
        self.fc1 = model_blocks.ConvBnReLu(
            in_channel=256*(len(atrous_rates)+2),
            out_channel=256,kernal_size=1,stride=1,padding=0,dilation=1
        )

        self.reduce = model_blocks.ConvBnReLu(width*4,48,kernal_size=1,stride=1,
                                              padding=0,dilation=1)
        self.fc2 = nn.Sequential(
            model_blocks.ConvBnReLu(304,256,kernal_size=3),
            model_blocks.ConvBnReLu(256,256,kernal_size=3),
            nn.Conv2d(256,n_class,kernel_size=1)
        )

    def forward(self, x):
        h = self.C(x)
        h = self.L1(h)
        h_1 = self.reduce(h)

        h = self.L2(h)
        h = self.L3(h)
        h = self.L4(h)
        h = self.aspp(h)
        h = self.fc1(h)
        h = F.interpolate(h,size=h_1.shape[2:],mode='bilinear',align_corners=False)

        h = torch.cat([h_1,h],dim=1)
        h = self.fc2(h)
        h = F.interpolate(h,size=x.shape[2:],mode='bilinear',align_corners=False)
        return h

class UnetAE(nn.Module):
    def __init__(self,in_channel,width,image_size,code_width):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel,width,kernel_size=3,padding=1),
            nn.Tanh(inplace=True),
            nn.Conv2d(width,width,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width,width*2,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width*2, width*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width*2, width * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 4, width * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.nnode = width*4*(image_size//4)*(image_size//4)

        self.Linear_encoder = nn.Linear(self.nnode,code_width)

        self.Linear_decoder = nn.Linear(code_width,)

        self.decoder = nn.Sequential(
            nn.Conv2d(width*4, width*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(width * 4, width * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(width * 2, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, in_channel, kernel_size=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

