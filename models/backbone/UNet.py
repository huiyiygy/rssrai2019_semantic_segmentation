import torch
import torch.nn as nn
from models.utils.layers import UnetConv2, UnetUp
from models.utils.utils import init_weights, count_param
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, feature_scale=2, is_deconv=True, sync_bn=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv

        if sync_bn:
            self.batchnorm = SynchronizedBatchNorm2d
        else:
            self.batchnorm = nn.BatchNorm2d

        filters = [64, 128, 256, 512, 1024]
        filters = [int(i / feature_scale) for i in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = UnetConv2(self.in_channels, filters[0], self.batchnorm)
        self.conv2 = UnetConv2(filters[0], filters[1], self.batchnorm)
        self.conv3 = UnetConv2(filters[1], filters[2], self.batchnorm)
        self.conv4 = UnetConv2(filters[2], filters[3], self.batchnorm)
        self.center = UnetConv2(filters[3], filters[4], self.batchnorm)
        self.dropout = nn.Dropout(0.5)
        # upsampling
        self.up_concat4 = UnetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UnetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UnetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UnetUp(filters[1], filters[0], self.is_deconv)
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        self._init_weight()

    def forward(self, inputs):
        conv1 = self.conv1(inputs)           # 16*512*512
        maxpool1 = self.maxpool(conv1)       # 16*256*256
        
        conv2 = self.conv2(maxpool1)         # 32*256*256
        maxpool2 = self.maxpool(conv2)       # 32*128*128

        conv3 = self.conv3(maxpool2)         # 64*128*128
        maxpool3 = self.maxpool(conv3)       # 64*64*64

        conv4 = self.conv4(maxpool3)         # 128*64*64
        maxpool4 = self.maxpool(conv4)       # 128*32*32

        center = self.center(maxpool4)       # 256*32*32
        center = self.dropout(center)

        up4 = self.up_concat4(center, conv4)  # 128*64*64
        up3 = self.up_concat3(up4, conv3)     # 64*128*128
        up2 = self.up_concat2(up3, conv2)     # 32*256*256
        up1 = self.up_concat1(up2, conv1)     # 16*512*512

        final = self.final(up1)

        return final

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.UpsamplingBilinear2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.UpsamplingBilinear2d) \
                    or isinstance(m, SynchronizedBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    if p.requires_grad:
                        yield p


if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable
    x = Variable(torch.rand(2, 1, 64, 64)).cuda()
    model = UNet().cuda()
    param = count_param(model)
    y = model(x)
    print('Output shape:', y.shape)
    print('UNet total parameters: %.2fM (%d)' % (param/1e6, param))
