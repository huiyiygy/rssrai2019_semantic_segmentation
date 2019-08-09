import torch
import torch.nn as nn
from models.utils.layers import UnetConv2, UnetUp
from models.utils.utils import count_param
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class UNetNested(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, feature_scale=2, is_deconv=True, is_ds=True, sync_bn=False):
        super(UNetNested, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_ds = is_ds

        if sync_bn:
            self.batchnorm = SynchronizedBatchNorm2d
        else:
            self.batchnorm = nn.BatchNorm2d

        filters = [64, 128, 256, 512, 1024]
        filters = [int(i / self.feature_scale) for i in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv00 = UnetConv2(self.in_channels, filters[0], self.batchnorm)
        self.conv10 = UnetConv2(filters[0], filters[1], self.batchnorm)
        self.conv20 = UnetConv2(filters[1], filters[2], self.batchnorm)
        self.conv30 = UnetConv2(filters[2], filters[3], self.batchnorm)
        self.conv40 = UnetConv2(filters[3], filters[4], self.batchnorm)

        # upsampling
        self.up_concat01 = UnetUp(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = UnetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = UnetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = UnetUp(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = UnetUp(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = UnetUp(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = UnetUp(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = UnetUp(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = UnetUp(filters[2], filters[1], self.is_deconv, 4)
        
        self.up_concat04 = UnetUp(filters[1], filters[0], self.is_deconv, 5)
        
        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        self._init_weight()

    def forward(self, inputs):
        # column : 0
        x_00 = self.conv00(inputs)       # 16*512*512
        maxpool0 = self.maxpool(x_00)    # 16*256*256
        x_10 = self.conv10(maxpool0)      # 32*256*256
        maxpool1 = self.maxpool(x_10)    # 32*128*128
        x_20 = self.conv20(maxpool1)     # 64*128*128
        maxpool2 = self.maxpool(x_20)    # 64*64*64
        x_30 = self.conv30(maxpool2)     # 128*64*64
        maxpool3 = self.maxpool(x_30)    # 128*32*32
        x_40 = self.conv40(maxpool3)     # 256*32*32
        # column : 1
        x_01 = self.up_concat01(x_10, x_00)
        x_11 = self.up_concat11(x_20, x_10)
        x_21 = self.up_concat21(x_30, x_20)
        x_31 = self.up_concat31(x_40, x_30)
        # column : 2
        x_02 = self.up_concat02(x_11, x_00, x_01)
        x_12 = self.up_concat12(x_21, x_10, x_11)
        x_22 = self.up_concat22(x_31, x_20, x_21)
        # column : 3
        x_03 = self.up_concat03(x_12, x_00, x_01, x_02)
        x_13 = self.up_concat13(x_22, x_10, x_11, x_12)
        # column : 4
        x_04 = self.up_concat04(x_13, x_00, x_01, x_02, x_03)

        # final layer
        final_1 = self.final_1(x_01)
        final_2 = self.final_2(x_02)
        final_3 = self.final_3(x_03)
        final_4 = self.final_4(x_04)

        final = (final_1+final_2+final_3+final_4)/4

        if self.is_ds:
            return final
        else:
            return final_4

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
    model = UNetNested().cuda()
    param = count_param(model)
    y = model(x)
    print('Output shape:', y.shape)
    print('UNet++ total parameters: %.2fM (%d)' % (param/1e6, param))
