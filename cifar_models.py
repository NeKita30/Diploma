import torch
import torch.nn as nn
import torch.nn.functional as F
from quantized_layers import FConv2d, QConv2d


class CNN6(nn.Module):
    def __init__(self, activation=F.relu6, quantized=False):
        super().__init__()
        self.act = activation
        # 3x32x32
        self.conv0 = FConv2d(3, 4, 1, pre_activation=None)
        # 4x32x32
        self.conv1 = FConv2d(4, 8, 5, pre_activation=F.hardtanh)
        self.pool1 = nn.MaxPool2d(2)
        # 8x14x14
        self.conv2 = FConv2d(8, 16, 3, pre_activation=self.act)
        self.pool2 = nn.MaxPool2d(2)
        # 16x6x6
        self.conv3 = FConv2d(16, 32, 3, pre_activation=self.act)
        self.pool3 = nn.MaxPool2d(2)
        # 32x2x2
        self.fc1 = FConv2d(32, 64, 2, pre_activation=self.act, bias=True, batch_norm=False)
        self.fc2 = nn.Linear(64, 10)

        if quantized:
            self.conv1 = QConv2d(self.conv1, quant_mode='void')
            self.conv2 = QConv2d(self.conv2, quant_mode='void')
            self.conv3 = QConv2d(self.conv3, quant_mode='void')
            self.fc1 = QConv2d(self.fc1, quant_mode='void')

        self.conv_layers = ["conv0", "conv1", "conv2", "conv3", "fc1"]

        self.pf_caches = [[], [], []]

    def forward(self, x):
        x = self.conv0(x) / 2
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = torch.tanh(self.fc1(x))
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        return x

    def cache(self, x):
        x = self.conv0(x) / 2
        x = self.pool1(self.conv1.cache(x))
        x = self.pool2(self.conv2.cache(x))
        x = self.pool3(self.conv3.cache(x))
        x = torch.tanh(self.fc1.cache(x))
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        return x

    def drop_cache(self):
        self.conv1.drop_cache()
        self.conv2.drop_cache()
        self.conv3.drop_cache()
        self.fc1.drop_cache()

    def cache_pooled_features(self, x):
        x = self.conv0(x) / 2
        x = self.pool1(self.conv1(x))
        self.pf_caches[0].append(x)
        x = self.pool2(self.conv2(x))
        self.pf_caches[1].append(x)
        x = self.pool3(self.conv3(x))
        self.pf_caches[2].append(x)
        x = torch.tanh(self.fc1(x))
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        return x

    def drop_pf_caches(self):
        self.pf_caches = [[], [], []]

    @staticmethod
    def layer_iterator():
        yield 'conv1'
        yield 'conv2'
        yield 'conv3'
        yield 'fc1'

    @staticmethod
    def freeze_layer_iterator():
        yield 'conv0'
        yield 'conv1'
        yield 'conv2'
        yield 'conv3'
        yield 'fc1'


class CNN7(nn.Module):
    def __init__(self, activation=F.relu6, quantized=False):
        super().__init__()
        self.act = activation
        # 3x32x32
        self.conv0 = FConv2d(3, 8, 1, pre_activation=None)
        # 4x32x32
        self.conv1_1 = FConv2d(8, 8, 3, pre_activation=F.hardtanh)
        self.conv1_2 = FConv2d(8, 12, 3, pre_activation=self.act)
        self.pool1 = nn.MaxPool2d(2)
        # 8x14x14
        self.conv2 = FConv2d(12, 16, 3, pre_activation=self.act)
        self.pool2 = nn.MaxPool2d(2)
        # 16x6x6
        self.conv3 = FConv2d(16, 32, 3, pre_activation=self.act)
        self.pool3 = nn.MaxPool2d(2)
        # 32x2x2
        self.fc1 = FConv2d(32, 64, 2, pre_activation=self.act, bias=True, batch_norm=False)
        self.fc2 = nn.Linear(64, 10)

        if quantized:
            self.conv1_1 = QConv2d(self.conv1_1, quant_mode='void')
            self.conv1_2 = QConv2d(self.conv1_2, quant_mode='void')
            self.conv2 = QConv2d(self.conv2, quant_mode='void')
            self.conv3 = QConv2d(self.conv3, quant_mode='void')
            self.fc1 = QConv2d(self.fc1, quant_mode='void')

        self.conv_layers = ["conv0", "conv1_1", "conv1_2", "conv2", "conv3", "fc1"]

        self.pf_caches = [[], [], []]

    def forward(self, x):
        x = self.conv0(x) / 2
        x = self.conv1_1(x)
        x = self.pool1(self.conv1_2(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = torch.tanh(self.fc1(x))
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        return x

    def cache(self, x):
        x = self.conv0(x) / 2
        x = self.conv1_1.cache(x)
        x = self.pool1(self.conv1_2.cache(x))
        x = self.pool2(self.conv2.cache(x))
        x = self.pool3(self.conv3.cache(x))
        x = torch.tanh(self.fc1.cache(x))
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        return x

    def drop_cache(self):
        self.conv1_1.drop_cache()
        self.conv1_2.drop_cache()
        self.conv2.drop_cache()
        self.conv3.drop_cache()
        self.fc1.drop_cache()

    def cache_pooled_features(self, x):
        x = self.conv0(x) / 2
        x = self.pool1(self.conv1_2(self.conv1_1(x)))
        self.pf_caches[0].append(x)
        x = self.pool2(self.conv2(x))
        self.pf_caches[1].append(x)
        x = self.pool3(self.conv3(x))
        self.pf_caches[2].append(x)
        x = torch.tanh(self.fc1(x))
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        return x

    def drop_pf_caches(self):
        self.pf_caches = [[], [], []]

    @staticmethod
    def layer_iterator():
        yield 'conv1_1'
        yield 'conv1_2'
        yield 'conv2'
        yield 'conv3'
        yield 'fc1'

    @staticmethod
    def freeze_layer_iterator():
        yield 'conv0'
        yield 'conv1_1'
        yield 'conv1_2'
        yield 'conv2'
        yield 'conv3'
        yield 'fc1'


class CNN8(nn.Module):
    def __init__(self, activation=F.relu6, quantized=False):
        super().__init__()
        self.act = activation
        # 3x32x32
        self.conv0 = FConv2d(3, 8, 1, pre_activation=None)
        # 4x32x32
        self.conv1_1 = FConv2d(8, 8, 3, pre_activation=F.hardtanh)
        self.conv1_2 = FConv2d(8, 12, 3, pre_activation=self.act)
        self.pool1 = nn.MaxPool2d(2)
        # 8x14x14
        self.conv2 = FConv2d(12, 24, 3, pre_activation=self.act)
        self.pool2 = nn.MaxPool2d(2)
        # 16x6x6
        self.conv3_1 = FConv2d(24, 24, 3, pre_activation=self.act)
        self.conv3_2 = FConv2d(24, 40, 3, pre_activation=self.act)
        # 32x2x2
        self.fc1 = FConv2d(40, 64, 2, pre_activation=self.act, bias=True, batch_norm=False)
        self.fc2 = nn.Linear(64, 10)

        if quantized:
            self.conv1_1 = QConv2d(self.conv1_1, quant_mode='void')
            self.conv1_2 = QConv2d(self.conv1_2, quant_mode='void')
            self.conv2 = QConv2d(self.conv2, quant_mode='void')
            self.conv3_1 = QConv2d(self.conv3_1, quant_mode='void')
            self.conv3_2 = QConv2d(self.conv3_2, quant_mode='void')
            self.fc1 = QConv2d(self.fc1, quant_mode='void')

        self.conv_layers = ["conv0", "conv1_1", "conv1_2", "conv2", "conv3_1", "conv3_2", "fc1"]

        self.pf_caches = [[], []]

    def forward(self, x):
        x = self.conv0(x) / 2
        x = self.conv1_1(x)
        x = self.pool1(self.conv1_2(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = torch.tanh(self.fc1(x))
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        return x

    def cache(self, x):
        x = self.conv0(x) / 2
        x = self.conv1_1.cache(x)
        x = self.pool1(self.conv1_2.cache(x))
        x = self.pool2(self.conv2.cache(x))
        x = self.conv3_1.cache(x)
        x = self.conv3_2.cache(x)
        x = torch.tanh(self.fc1.cache(x))
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        return x

    def drop_cache(self):
        self.conv1_1.drop_cache()
        self.conv1_2.drop_cache()
        self.conv2.drop_cache()
        self.conv3_1.drop_cache()
        self.conv3_2.drop_cache()
        self.fc1.drop_cache()

    def cache_pooled_features(self, x):
        x = self.conv0(x) / 2
        x = self.pool1(self.conv1_2(self.conv1_1(x)))
        self.pf_caches[0].append(x)
        x = self.pool2(self.conv2(x))
        self.pf_caches[1].append(x)
        x = self.conv3_2(self.conv3_1(x))
        x = torch.tanh(self.fc1(x))
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        return x

    def drop_pf_caches(self):
        self.pf_caches = [[], []]

    @staticmethod
    def layer_iterator():
        yield 'conv1_1'
        yield 'conv1_2'
        yield 'conv2'
        yield 'conv3_1'
        yield 'conv3_2'
        yield 'fc1'

    @staticmethod
    def freeze_layer_iterator():
        yield 'conv0'
        yield 'conv1_1'
        yield 'conv1_2'
        yield 'conv2'
        yield 'conv3_1'
        yield 'conv3_2'
        yield 'fc1'


class CNN9(nn.Module):
    def __init__(self, activation=F.relu6, quantized=False):
        super().__init__()
        self.act = activation
        # 3x32x32
        self.conv0 = FConv2d(3, 8, 1, pre_activation=None)
        # 4x32x32
        self.conv1_1 = FConv2d(8, 8, 3, pre_activation=F.hardtanh)
        self.conv1_2 = FConv2d(8, 12, 3, pre_activation=self.act)
        self.pool1 = nn.MaxPool2d(2)
        # 8x14x14
        self.conv2_1 = FConv2d(12, 12, 3, padding=1, pre_activation=self.act)
        self.conv2_2 = FConv2d(12, 24, 3, pre_activation=self.act)
        self.pool2 = nn.MaxPool2d(2)
        # 16x6x6
        self.conv3_1 = FConv2d(24, 24, 3, pre_activation=self.act)
        self.conv3_2 = FConv2d(24, 48, 3, pre_activation=self.act)
        # 32x2x2
        self.fc1 = FConv2d(48, 96, 2, pre_activation=self.act, bias=True, batch_norm=False)
        self.fc2 = nn.Linear(96, 10)

        if quantized:
            self.conv1_1 = QConv2d(self.conv1_1, quant_mode='void')
            self.conv1_2 = QConv2d(self.conv1_2, quant_mode='void')
            self.conv2_1 = QConv2d(self.conv2_1, quant_mode='void')
            self.conv2_2 = QConv2d(self.conv2_2, quant_mode='void')
            self.conv3_1 = QConv2d(self.conv3_1, quant_mode='void')
            self.conv3_2 = QConv2d(self.conv3_2, quant_mode='void')
            self.fc1 = QConv2d(self.fc1, quant_mode='void')

        self.conv_layers = ["conv0", "conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "fc1"]

        self.pf_caches = [[], []]

    def forward(self, x):
        x = self.conv0(x) / 2
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = torch.tanh(self.fc1(x))
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        return x

    def cache(self, x):
        x = self.conv0(x) / 2
        x = self.conv1_1.cache(x)
        x = self.conv1_2.cache(x)
        x = self.pool1(x)
        x = self.conv2_1.cache(x)
        x = self.conv2_2.cache(x)
        x = self.pool2(x)
        x = self.conv3_1.cache(x)
        x = self.conv3_2.cache(x)
        x = torch.tanh(self.fc1.cache(x))
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        return x

    def drop_cache(self):
        self.conv1_1.drop_cache()
        self.conv1_2.drop_cache()
        self.conv2_1.drop_cache()
        self.conv2_2.drop_cache()
        self.conv3_1.drop_cache()
        self.conv3_2.drop_cache()
        self.fc1.drop_cache()

    def cache_pooled_features(self, x):
        x = self.conv0(x) / 2
        x = self.pool1(self.conv1_2(self.conv1_1(x)))
        self.pf_caches[0].append(x)
        x = self.pool2(self.conv2_2(self.conv2_1(x)))
        self.pf_caches[1].append(x)
        x = self.conv3_2(self.conv3_1(x))
        x = torch.tanh(self.fc1(x))
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        return x

    def drop_pf_caches(self):
        self.pf_caches = [[], []]

    @staticmethod
    def layer_iterator():
        yield 'conv1_1'
        yield 'conv1_2'
        yield 'conv2_1'
        yield 'conv2_2'
        yield 'conv3_1'
        yield 'conv3_2'
        yield 'fc1'

    @staticmethod
    def freeze_layer_iterator():
        yield 'conv0'
        yield 'conv1_1'
        yield 'conv1_2'
        yield 'conv2_1'
        yield 'conv2_2'
        yield 'conv3_1'
        yield 'conv3_2'
        yield 'fc1'


class CNN10(nn.Module):
    def __init__(self, activation=F.relu6, quantized=False):
        super().__init__()
        self.act = activation
        # 3x32x32
        self.conv0 = FConv2d(3, 8, 1, pre_activation=None)
        # 8x32x32
        self.conv1_1 = FConv2d(8, 16, 3, padding=1, pre_activation=F.hardtanh)
        self.conv1_2 = FConv2d(16, 32, 3, padding=1, pre_activation=self.act)
        self.pool1 = nn.MaxPool2d(2)
        # 32x16x16
        self.conv2_1 = FConv2d(32, 32, 3, padding=1, pre_activation=self.act)
        self.conv2_2 = FConv2d(32, 64, 3, padding=1, pre_activation=self.act)
        self.pool2 = nn.MaxPool2d(2)
        # 64x8x8
        self.conv3_1 = FConv2d(64, 64, 3, pre_activation=self.act)
        self.conv3_2 = FConv2d(64, 64, 3, pre_activation=self.act)
        self.conv3_3 = FConv2d(64, 128, 3, pre_activation=self.act)
        # 32x2x2
        self.fc1 = FConv2d(128, 256, 2, pre_activation=self.act, bias=True, batch_norm=False)
        self.fc2 = nn.Linear(256, 10)

        if quantized:
            self.conv1_1 = QConv2d(self.conv1_1, quant_mode='void')
            self.conv1_2 = QConv2d(self.conv1_2, quant_mode='void')
            self.conv2_1 = QConv2d(self.conv2_1, quant_mode='void')
            self.conv2_2 = QConv2d(self.conv2_2, quant_mode='void')
            self.conv3_1 = QConv2d(self.conv3_1, quant_mode='void')
            self.conv3_2 = QConv2d(self.conv3_2, quant_mode='void')
            self.conv3_3 = QConv2d(self.conv3_3, quant_mode='void')
            self.fc1 = QConv2d(self.fc1, quant_mode='void')

        self.conv_layers = ["conv0", "conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "conv3_3", "fc1"]

        self.pf_caches = [[], []]

    def forward(self, x):
        x = self.conv0(x) / 2
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = torch.tanh(self.fc1(x))
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        return x

    def cache(self, x):
        x = self.conv0(x) / 2
        x = self.conv1_1.cache(x)
        x = self.conv1_2.cache(x)
        x = self.pool1(x)
        x = self.conv2_1.cache(x)
        x = self.conv2_2.cache(x)
        x = self.pool2(x)
        x = self.conv3_1.cache(x)
        x = self.conv3_2.cache(x)
        x = self.conv3_3.cache(x)
        x = torch.tanh(self.fc1.cache(x))
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        return x

    def drop_cache(self):
        self.conv1_1.drop_cache()
        self.conv1_2.drop_cache()
        self.conv2_1.drop_cache()
        self.conv2_2.drop_cache()
        self.conv3_1.drop_cache()
        self.conv3_2.drop_cache()
        self.conv3_3.drop_cache()
        self.fc1.drop_cache()

    def cache_pooled_features(self, x):
        x = self.conv0(x) / 2
        x = self.pool1(self.conv1_2(self.conv1_1(x)))
        self.pf_caches[0].append(x)
        x = self.pool2(self.conv2_2(self.conv2_1(x)))
        self.pf_caches[1].append(x)
        x = self.conv3_3(self.conv3_2(self.conv3_1(x)))
        x = torch.tanh(self.fc1(x))
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        return x

    def drop_pf_caches(self):
        self.pf_caches = [[], []]

    @staticmethod
    def layer_iterator():
        yield 'conv1_1'
        yield 'conv1_2'
        yield 'conv2_1'
        yield 'conv2_2'
        yield 'conv3_1'
        yield 'conv3_2'
        yield 'conv3_3'
        yield 'fc1'

    @staticmethod
    def freeze_layer_iterator():
        yield 'conv0'
        yield 'conv1_1'
        yield 'conv1_2'
        yield 'conv2_1'
        yield 'conv2_2'
        yield 'conv3_1'
        yield 'conv3_2'
        yield 'conv3_3'
        yield 'fc1'


def get_model(model: str, quantized=False):
    if model == 'CNN6':
        network = CNN6(quantized=quantized)
    elif model == 'CNN7':
        network = CNN7(quantized=quantized)
    elif model == 'CNN8':
        network = CNN8(quantized=quantized)
    elif model == 'CNN9':
        network = CNN9(quantized=quantized)
    elif model == 'CNN10':
        network = CNN10(quantized=quantized)
    else:
        raise RuntimeError('Incorrect model')
    return network
