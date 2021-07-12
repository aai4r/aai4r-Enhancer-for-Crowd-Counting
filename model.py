import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, \
                stride=1, NL='relu', same_padding=False, bn=False, dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        self.conv = []
        if dilation==1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else nn.Identity()
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, sync=False, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        if sync:
            # for sync bn
            print('use sync inception')
            self.bn = nn.SyncBatchNorm(out_channels, eps=0.001)
        else:
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class PFSNet(nn.Module):
    def __init__(self, pretrained=False, args=None):
        super(PFSNet, self).__init__()
        
        # docoder definition
        self.de_pred5 = nn.Sequential(
            Conv2d(512, 1024, 3, same_padding=True, NL='relu'),
            Conv2d(1024, 512, 3, same_padding=True, NL='relu'),
        )

        self.de_pred4 = nn.Sequential(
            Conv2d(512 + 512, 512, 3, same_padding=True, NL='relu'),
            Conv2d(512, 256, 3, same_padding=True, NL='relu'),
        )

        self.de_pred3 = nn.Sequential(
            Conv2d(256 + 256, 256, 3, same_padding=True, NL='relu'),
            Conv2d(256, 128, 3, same_padding=True, NL='relu'),
        )

        self.de_pred2 = nn.Sequential(
            Conv2d(128 + 128, 128, 3, same_padding=True, NL='relu'),
            Conv2d(128, 64, 3, same_padding=True, NL='relu'),
        )

        self.de_pred1 = nn.Sequential(
            Conv2d(64 + 64, 64, 3, same_padding=True, NL='relu'),
            Conv2d(64, 64, 3, same_padding=True, NL='relu'),
        )
        self.in_channels = 64
        sync = False
        self.density_head5 = nn.Sequential(
            TwoBranchModule(512, sync=sync),
            Conv2d(1024, self.in_channels, 1, same_padding=True)
        )

        self.density_head4 = nn.Sequential(
            TwoBranchModule(256, sync=sync),
            Conv2d(512, self.in_channels, 1, same_padding=True)
        )

        self.density_head3 = nn.Sequential(
            TwoBranchModule(128, sync=sync),
            Conv2d(256, self.in_channels, 1, same_padding=True)
        )

        self.density_head2 = nn.Sequential(
            TwoBranchModule(64, sync=sync),
            Conv2d(128, self.in_channels, 1, same_padding=True)
        )

        self.density_head1 = nn.Sequential(
            TwoBranchModule(64, sync=sync),
            Conv2d(128, self.in_channels, 1, same_padding=True)
        )
        
        d = self.in_channels // args.reduction
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
                        nn.Conv2d(self.in_channels, d, kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm2d(d),
                        nn.ReLU(inplace=False)
        )
        self.M = 5
        self.fcs = nn.ModuleList([])
        for i in range(self.M):
            self.fcs.append(
                 nn.Conv2d(d, self.in_channels, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)
        self.conv = nn.Conv2d(self.in_channels, 1, kernel_size=1, stride=1)
        self._init_weight()
        
        # define the backbone network
        vgg = models.vgg16_bn(pretrained=pretrained)
        print("pretrained vgg? ", pretrained)
        features = list(vgg.features.children())
        # get each stage of the backbone
        self.features1 = nn.Sequential(*features[0:6])
        self.features2 = nn.Sequential(*features[6:13])
        self.features3 = nn.Sequential(*features[13:23])
        self.features4 = nn.Sequential(*features[23:33])
        self.features5 = nn.Sequential(*features[33:43])

    # the forward process
    def forward(self, x):
        size = x.size()
        x1 = self.features1(x)
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)
        # begining of decoding
        x = self.de_pred5(x5)
        x5_out = x
        x = F.upsample_bilinear(x, size=x4.size()[2:])

        x = torch.cat([x4, x], 1)
        x = self.de_pred4(x)
        x4_out = x
        x = F.upsample_bilinear(x, size=x3.size()[2:])

        x = torch.cat([x3, x], 1)
        x = self.de_pred3(x)
        x3_out = x
        x = F.upsample_bilinear(x, size=x2.size()[2:])

        x = torch.cat([x2, x], 1)
        x = self.de_pred2(x)
        x2_out = x
        x = F.upsample_bilinear(x, size=x1.size()[2:])

        x = torch.cat([x1, x], 1)
        x = self.de_pred1(x)
        x1_out = x
        # density prediction
        x5_density = self.density_head5(x5_out)
        x4_density = self.density_head4(x4_out)
        x3_density = self.density_head3(x3_out)
        x2_density = self.density_head2(x2_out)
        x1_density = self.density_head1(x1_out)
        
        # upsample the density prediction to be the same with the input size
        x5_density = F.upsample_nearest(x5_density, size=x1.size()[2:])
        x4_density = F.upsample_nearest(x4_density, size=x1.size()[2:])
        x3_density = F.upsample_nearest(x3_density, size=x1.size()[2:])
        x2_density = F.upsample_nearest(x2_density, size=x1.size()[2:])
        x1_density = F.upsample_nearest(x1_density, size=x1.size()[2:])
        
        density_map = torch.cat([x5_density, x4_density, x3_density, x2_density, x1_density], 1)
        density_map_ = density_map.view(size[0], self.M, 64, density_map.shape[2], density_map.shape[3])
        
        u = torch.sum(density_map_, dim=1)
        s = self.gap(u)
        z = self.fc(s)
        
        attention_vectors = [fc(z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(size[0], self.M, 64, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        density = torch.sum(density_map_*attention_vectors, dim=1)
        density = self.conv(density)
        x1_density = self.conv(x1_density)
        x2_density = self.conv(x2_density)
        x3_density = self.conv(x3_density)
        x4_density = self.conv(x4_density)
        x5_density = self.conv(x5_density)
        return density, x1_density, x2_density, x3_density, x4_density, x5_density

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class TwoBranchModule(nn.Module):
    def __init__(self, in_channels, sync=False):
        super(TwoBranchModule, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, in_channels//2, kernel_size=1, sync=sync)
        self.branch3x3_2 = BasicConv2d(in_channels // 2, in_channels, kernel_size=(3, 3), padding=(1, 1), sync=sync)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        outputs = [branch3x3, x]
        return torch.cat(outputs, 1)


class Enhancer(nn.Module):
    def __init__(self):
        super(Enhancer, self).__init__()
        self.down1 = nn.Conv2d(kernel_size=3, padding=1, in_channels=3, out_channels=64)
        self.down2 = nn.Conv2d(kernel_size=3, padding=1, in_channels=64, out_channels=128)
        self.down3 = nn.Conv2d(kernel_size=3, padding=1, in_channels=128, out_channels=256)
        self.up1 = nn.Conv2d(kernel_size=3, padding=1, in_channels=256, out_channels=128)
        self.up2 = nn.Conv2d(kernel_size=3, padding=1, in_channels=128, out_channels=64)
        self.up3 = nn.Conv2d(kernel_size=3, padding=1, in_channels=64, out_channels=3)
        self.up4 = nn.Conv2d(kernel_size=3, padding=1, in_channels=64, out_channels=3)
        self.down = nn.MaxPool2d(4)
        self.up = nn.Upsample(scale_factor=4, mode='nearest')

    def forward(self, x):
        input_size = x.size()[2:]
        x = self.down1(x)
        x = torch.relu(x)
        x1 = self.down(x)
        x = self.down2(x1)
        x = torch.relu(x)
        x2 = self.down(x)
        x = self.down3(x2)
        x = torch.relu(x)
        x = F.upsample_nearest(x, size=x1.size()[2:])
        x = self.up1(x)
        x = torch.relu(x)
        x = self.up2(F.upsample_nearest(x, size=input_size))
        x = torch.relu(x)
        beta = self.up3(x)
        gamma = self.up4(x)

        beta = torch.sigmoid(beta)*2-1
        gamma = torch.sigmoid(gamma)*4
        return beta, gamma
