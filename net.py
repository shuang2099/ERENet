import torch
import torch.nn.parallel
import torch.optim
import torch.nn.init as init
import torch.nn as nn
from fusion import CBAM


# from prenet import *
def initialize_weights(self):
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight, gain=1.0)
            if m.weight.data.shape[1] == 1:
                init.normal_(m.weight, mean=0.0, std=0.1)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight, gain=1.0)
            if m.weight.data.shape[1] == 1:
                init.normal_(m.weight, std=0.1)
            if m.bias is not None:
                init.zeros_(m.bias)
    self.apply(weight_init)


class cov1(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel=1, use_bn=True, use_relu=True, padding=0):
        super(cov1, self).__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu
        self.cov1 = nn.Conv2d(inp_dim, out_dim, stride=1, kernel_size=kernel, padding=padding)
        self.bn = nn.BatchNorm2d(out_dim, affine=True)
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        out = self.cov1(x)
        if self.use_bn:
            out = self.bn(out)
        if self.use_relu:
            out = self.relu(out)
        return out


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, int(out_dim / 2), 3, 1, 2)
        self.bn1 = nn.BatchNorm2d( int(out_dim / 2), affine=True)
        self.conv2 = nn.Conv2d(int(out_dim / 2), out_dim, 3, 1, 0)
        self.bn2 = nn.BatchNorm2d(out_dim,affine=True)
        self.relu = nn.LeakyReLU(inplace=True)
        self.skip_layer = nn.Conv2d(inp_dim, out_dim, 1, 1, 0)

        if inp_dim == out_dim: self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):

        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = out + residual
        return out


class prenet2(nn.Module):
    def __init__(self, channel=128):
        super(prenet2, self).__init__()
        self.res1 = Residual(3, channel * 2)
        self.res2 = Residual(channel * 2, channel)
        self.res3 = Residual(channel, 32)

    def forward(self, x):
        out = self.res1(x)
        out = self.res2(out)
        out = self.res3(out)

        return out

class backbone(nn.Module):
    def __init__(self):
        super(backbone, self).__init__()

        self.encorder1 = nn.Sequential(
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                        Residual(32, 64, ))

        self.encorder2 = nn.Sequential(
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                        Residual(64, 256, ))

        self.encorder3 = nn.Sequential(
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                        Residual(256, 512))

        self.encorder4 = nn.Sequential(
                        Residual(512, 512, ))


        self.decorder1 = nn.Sequential(
                         nn.Upsample(scale_factor=2, mode='bilinear',),
                         Residual(512, 256))

        self.decorder2 = nn.Sequential(
                         nn.Upsample(scale_factor=2, mode='bilinear',),
                         Residual(256,64))

        self.decorder3 = nn.Sequential(
                         nn.Upsample(scale_factor=2, mode='bilinear',),
                         Residual(64, 32))

    def forward(self, x):

        encorder1 = self.encorder1(x)
        encorder2 = self.encorder2(encorder1)
        encorder3 = self.encorder3(encorder2)
        encorder4 = self.encorder4(encorder3)
        decorder1 = self.decorder1(encorder4+encorder3)
        decorder2 = self.decorder2(decorder1+encorder2)
        decorder3 = self.decorder3(decorder2+encorder1)

        return decorder3
class field(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride,padding):
        super(field, self).__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        self.bn = nn.BatchNorm2d(out_dim, affine=True)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        bn = self.bn(conv2)
        out = self.relu(bn)
        return out
class fusionblock1(nn.Module):
    def __init__(self,):
        super(fusionblock1, self).__init__()

        self.field11 = field(4, 128, 3, 1,  1)
        self.field12 = field(4, 128, 7, 1, 3)
        self.field13 = field(4, 128, 11, 1, 5)
        self.field14 = field(4, 128, 15, 1, 7)

        self.field21 = field(128, 128, 3, 1, 1)
        self.field22 = field(128, 128, 7, 1, 3)
        self.field23 = field(128, 128, 11, 1, 5)
        self.field24 = field(128, 128, 15, 1, 7)


        self.field31 = field(128, 8, 3, 1, 1)
        self.field32 = field(128, 8, 7, 1, 3)
        self.field33 = field(128, 8, 11, 1, 5)
        self.field34 = field(128, 8, 15, 1, 7)



    def forward(self, x):

        field11 = self.field11(x)
        field12 = self.field12(x)
        field13 = self.field13(x)
        field14 = self.field14(x)


        field21 = self.field21(field11)
        field12 = field12+field11
        field22 = self.field22(field12)
        field13 = field13+field12
        field23 = self.field23(field13)
        field14 = field14+field13
        field24 = self.field24(field14)

        field21 = field21+field11+field22
        field31 = self.field31(field21)
        field22 = field22+field12+field23
        field32 = self.field32(field22)
        field23 = field23 + field13 + field24
        field33 = self.field33(field23)
        field24 = field24 + field14
        field34 = self.field34(field24)

        output = torch.cat((field31, field32, field33, field34),1)

        return output


class fusionblock2(nn.Module):
    def __init__(self):
        super(fusionblock2, self).__init__()
        self.res = Residual(32, 256, )

        self.CBAM1 = CBAM(256)
        self.CBAM2 = CBAM(256)
        self.res1 = Residual(256, 128, )
        self.res2 = Residual(256, 128, )
        self.r_res1 = Residual(256, 128, )
        self.r_res2 = Residual(128, 32, )
        self.cov1 = cov1(256, 32, kernel=1, padding=0)
        self.outres = Residual(32, 128, )
        self.outcov1 = cov1(128, 1, use_relu=False)


    def forward(self, x):
        res = self.res(x)
        CBAM1 = self.CBAM1(res)
        res1 = self.res1(CBAM1)
        CBAM2 = self.CBAM2(res)
        res2 = self.res2(CBAM2)
        fuse = torch.cat((res1, res2,), 1)
        fuse = self.cov1(fuse)
        r_res1 = self.r_res1(res)
        r_res2 = self.r_res2(r_res1)
        fuseout = (fuse * r_res2) + x
        output = self.outres(fuseout)
        output = self.outcov1(output)

        return output



class mynet(nn.Module):
    def __init__(self,):
        super(mynet, self).__init__()

        self.prenet2 = prenet2()
        self.backbone1 = backbone()
        self.backbone2 = backbone()
        self.backbone3 = backbone()

        self.covout1 = cov1(32, 1, use_relu=False)
        self.covout2 = cov1(32, 1, use_relu=False)
        self.covout3 = cov1(32, 1, use_relu=False)
        self.covout4 = cov1(32, 1, use_relu=False)
        self.covout5 = cov1(32, 1, use_relu=False)

        self.skip = cov1(4, 32,)

        self.fusionblock1 = fusionblock1()
        self.fusionblock2 = fusionblock2()

        self.apply(initialize_weights)

    def forward(self, x):
        in1 = self.prenet2(x)
        f1 = self.covout1(in1)

        backbone1 = self.backbone1(in1)
        in2 = backbone1 + in1
        f2 = self.covout2(backbone1)

        backbone2 = self.backbone2(in2)
        f3 = self.covout3(backbone2)

        in3 = backbone2 * backbone1
        backbone3 = self.backbone3(in3)

        in4 = backbone3 * backbone1
        f4 = self.covout4(in4)
        out = torch.cat((f1, f2, f3, f4), 1)

        fout1 = self.fusionblock1(out)
        f5 = self.covout5(fout1)

        skip = self.skip(out)

        fout2 = self.fusionblock2(fout1+skip)
        results = [f1, f2, f3, f4, f5, fout2]
        results = [torch.sigmoid(r) for r in results]

        return results