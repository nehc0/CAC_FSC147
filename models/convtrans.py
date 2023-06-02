import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import collections
from models.transformer_module import Transformer, PatchProj
from models.convolution_module import ConvBlock


class VGG16Trans(nn.Module):
    def __init__(self, batch_norm=True, load_weights=False):
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBlock(cin=3, cout=64),
            ConvBlock(cin=64, cout=64),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(cin=64, cout=128),
            ConvBlock(cin=128, cout=128),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(cin=128, cout=256),
            ConvBlock(cin=256, cout=256),
            ConvBlock(cin=256, cout=256),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.stage4 = nn.Sequential(
            ConvBlock(cin=256, cout=512),
            ConvBlock(cin=512, cout=512),
            ConvBlock(cin=512, cout=512),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvBlock(cin=512, cout=512),
            ConvBlock(cin=512, cout=512),
            ConvBlock(cin=512, cout=512),
        )
        self.vgg16 = nn.Sequential(
            self.stage1,
            self.stage2,
            self.stage3,
            self.stage4,
        )

        self.patchpj1 = PatchProj(kernel_size=16, input_size=16*16*3)
        self.patchpj2 = PatchProj(kernel_size=8, input_size=8*8*64)
        self.patchpj3 = PatchProj(kernel_size=4, input_size=4*4*128)
        self.patchpj4 = PatchProj(kernel_size=2, input_size=2*2*256)
        self.patchpj5 = PatchProj(kernel_size=1, input_size=1*1*512)

        self.conv00 = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1)
        self.conv01 = nn.Conv2d(in_channels=65, out_channels=64, kernel_size=1)
        self.conv02 = nn.Conv2d(in_channels=129, out_channels=128, kernel_size=1)
        self.conv03 = nn.Conv2d(in_channels=257, out_channels=256, kernel_size=1)
        self.conv04 = nn.Conv2d(in_channels=513, out_channels=512, kernel_size=1)

        self.encoder0 = Transformer(layers=1, dim=512, nhead=1)
        self.encoder1 = Transformer(layers=1, dim=512, nhead=1)
        self.encoder2 = Transformer(layers=1, dim=512, nhead=1)
        self.encoder3 = Transformer(layers=1, dim=512, nhead=1)
        self.encoder4 = Transformer(layers=1, dim=512, nhead=1)

        self.conv2 = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)

        self.transformer = Transformer(layers=4, dim=512, nhead=1)

        self.fold = nn.Fold(output_size=(192, 192), kernel_size=(8, 8), stride=8)

        self.conv3 = nn.Sequential(
                ConvBlock(cin=8, cout=4, k_size=3),
                nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1),
            )

        self._initialize_weights()
        if not load_weights:
            if batch_norm:
                mod = torchvision.models.vgg16_bn(pretrained=True)
            else:
                mod = torchvision.models.vgg16(pretrained=True)
            fsd = collections.OrderedDict()
            for i in range(len(self.vgg16.state_dict().items())):
                temp_key = list(self.vgg16.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.vgg16.load_state_dict(fsd)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, shot, scale_embed):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        b, c, h, w = x4.shape

        k1 = self.stage1(shot)
        k2 = self.stage2(k1)
        k3 = self.stage3(k2)
        k4 = self.stage4(k3)

        x0p = self.patchpj1(x)
        x1p = self.patchpj2(x1)
        x2p = self.patchpj3(x2)
        x3p = self.patchpj4(x3)
        x4p = self.patchpj5(x4)

        se0 = scale_embed.view(b, 1, 64, 64)
        se1 = F.avg_pool2d(se0, 2)
        se2 = F.avg_pool2d(se1, 2)
        se3 = F.avg_pool2d(se2, 2)
        se4 = F.avg_pool2d(se3, 2)

        k0se = torch.cat((shot, se0), dim=-3)
        k0 = self.conv00(k0se)
        k1se = torch.cat((k1, se1), dim=-3)
        k1 = self.conv01(k1se)
        k2se = torch.cat((k2, se2), dim=-3)
        k2 = self.conv02(k2se)
        k3se = torch.cat((k3, se3), dim=-3)
        k3 = self.conv03(k3se)
        k4se = torch.cat((k4, se4), dim=-3)
        k4 = self.conv04(k4se)

        k0p = self.patchpj1(k0)
        k1p = self.patchpj2(k1)
        k2p = self.patchpj3(k2)
        k3p = self.patchpj4(k3)
        k4p = self.patchpj5(k4)

        a0 = torch.cat((x0p, k0p), dim=-2).transpose(0, 1)
        a1 = torch.cat((x1p, k1p), dim=-2).transpose(0, 1)
        a2 = torch.cat((x2p, k2p), dim=-2).transpose(0, 1)
        a3 = torch.cat((x3p, k3p), dim=-2).transpose(0, 1)
        a4 = torch.cat((x4p, k4p), dim=-2).transpose(0, 1)

        a0 = self.encoder0(a0).transpose(0, 1).view(b, 1, h*w+4*4, c)
        a1 = self.encoder1(a1).transpose(0, 1).view(b, 1, h*w+4*4, c)
        a2 = self.encoder2(a2).transpose(0, 1).view(b, 1, h*w+4*4, c)
        a3 = self.encoder3(a3).transpose(0, 1).view(b, 1, h*w+4*4, c)
        a4 = self.encoder4(a4).transpose(0, 1).view(b, 1, h*w+4*4, c)

        tokens = torch.cat((a0, a1, a2, a3, a4), dim=1)
        tokens = self.conv2(tokens)
        tokens = tokens.view(b, 24*24+4*4, 512)  # -> b hw c

        x = tokens.transpose(0, 1)  # hw b c
        x = self.transformer(x)
        x = x.permute(1, 2, 0)  # b c hw
        assert list(x.size()) == [b, c, h*w+4*4]
        x_body, _ = x.split([h*w, 4*4], -1)
        assert list(x_body.size()) == [b, c, h*w]
        x = x_body
        x = self.fold(x)
        assert list(x.size()) == [b, 8, 192, 192]
        y = self.conv3(x)

        return y
