import torch.nn as nn
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


# initialize the module
def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def weights_init_kaiming(m):
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
        # nn.init.normal_(m.weight.data, 1.0, 0.02)
        m.weight.data.fill_(1)
        m.bias.data.zero_()


# compute model params
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count
