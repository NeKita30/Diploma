import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from quantization_utils import LinearQuant, LinearQuantShifted, Dequant, fake_quant, scale_quant

class FConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, groups=1,
                 batch_norm=True, pre_activation=None):
        super().__init__()
        if batch_norm:
            bias = False
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.pre_act = pre_activation
        self.bn = None
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        self.cache_mode = False
        self.cached_in = []
        self.cached_out = []

    def get_weight(self):
        return self.conv.weight

    def forward(self, x):
        if self.cache_mode:
            return self.cache(x)
        if self.pre_act is not None:
            x = self.pre_act(x)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x

    def cache(self, x):
        if self.pre_act is not None:
            x = self.pre_act(x)
        self.cached_in.append(torch.clone(x.detach()))
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        self.cached_out.append(torch.clone(x.detach()))
        return x

    def drop_cache(self):
        self.cached_in = []
        self.cached_out = []




class QConv2d(nn.Module):
    def __init__(self, fconv: FConv2d, in_bins=256, w_bins=256, symetric_mode=True, in_step=-1, quant_mode='min_max',
                 corrected_cached_in=None):
        super().__init__()
        device = fconv.conv.weight.device
        self.in_channels = fconv.conv.in_channels
        self.out_channels = fconv.conv.out_channels
        # wrapping batch norm into weights
        bn_scale = torch.ones((self.out_channels,), dtype=torch.float32, device=device)
        bias = torch.zeros((self.out_channels,), dtype=torch.float32, device=device)
        if fconv.conv.bias is not None:
            bias = fconv.conv.bias.data.clone()
        if fconv.bn is not None:
            bn = fconv.bn
            bn_scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            bias = (bias - bn.running_mean) * bn_scale + bn.bias

        self.f32_weights = nn.Parameter(fconv.conv.weight.data * (bn_scale.view((self.out_channels, 1, 1, 1))))
        self.f32_bias = nn.Parameter(bias)
        self.padding = fconv.conv.padding
        self.stride = fconv.conv.stride
        self.groups = fconv.conv.groups

        if isinstance(in_step, torch.Tensor):
            in_step = in_step.item()
        quantizer = None
        if quant_mode == 'min_max':
            quantizer = self._min_max_quant
        elif quant_mode == 'ada':
            quantizer = self._ada_quant
        elif quant_mode == 'void':
            self.input_quant = LinearQuant(in_bins, 1, 0)
            self.weight_quant = LinearQuant(w_bins, 1, 0)
            return
        else:
            raise RuntimeError(f'Invalid mode {quant_mode}')
        cached_in = fconv.cached_in if corrected_cached_in is None else corrected_cached_in
        quantizer(cached_in, fconv.cached_out, in_bins, w_bins, symetric_mode, in_step)

    def get_weight(self):
        return self.f32_weights

    def _min_max_quant(self, cached_in, cached_out, in_bins, w_bins, symetric_mode, in_step):
        # quantizing inputs
        in_range = [0, 0]
        if not cached_in:
            raise RuntimeError('cant calibrate input')
        for x in cached_in:
            in_range[0] = min(in_range[0], x.min().item())
            in_range[1] = max(in_range[1], x.max().item())
        in_scale = (in_range[1] - in_range[0]) / in_bins
        if in_step > 1e-9:
            in_scale = np.round(in_scale / in_step) * in_step
        in_offset = np.int32(np.round(-in_range[0] / in_scale))
        # quantizing weights
        if symetric_mode:
            if w_bins % 2 != 1:
                raise RuntimeError('Symetric requiers odd ammoint of w_bins')
            a = torch.max(torch.abs(self.f32_weights.data)).item()
            w_scale = 2 * a / w_bins
            w_offset = w_bins // 2
        else:
            r = torch.max(self.f32_weights.data).item()
            l = torch.min(self.f32_weights.data).item()
            w_scale = (r - l) / in_bins
            w_offset = np.int32(np.round(-l / w_scale))
        # initializing quantizaed params
        self.input_quant = LinearQuant(in_bins, in_scale, in_offset)
        self.weight_quant = LinearQuant(w_bins, w_scale, w_offset)

    def _ada_quant(self, cached_in, cached_out, in_bins, w_bins, symetric_mode, in_step):
        # HIST_BINS = 10000
        QUANTILE = 0.0001  # inintial scaling on range [0.1% quantile -- 99.9% quantile]
        in_range = [0, 0]
        if not cached_in:
            raise RuntimeError('cant calibrate input')
        for x in cached_in:
            in_range[0] = min(in_range[0], x.min().item())
            in_range[1] = max(in_range[1], x.max().item())

        # h_rscale = HIST_BINS / (in_range[1] - in_range[0])
        # h_offset = np.int32(np.round(-in_range[0] * h_rscale))
        # collecting histogram
        # hist = torch.zeros([HIST_BINS]).cpu()
        # stds = []
        # for x in cached_in:
        #     stds.append(x.std().item())
        #     signals = ((x.view([-1]).cpu() * h_rscale).long() + h_offset).clip(0, HIST_BINS - 1)
        #     idx, counts = torch.unique(signals, return_counts=True)
        #     hist[idx] += counts
        # hist /= torch.sum(hist)
        #
        # selected = [False, False]
        # accumulative_hist = hist.clone()
        #
        # for j in range(HIST_BINS):
        #     if j > 0:
        #         accumulative_hist[j] += accumulative_hist[j - 1]
        #     if (not selected[0]) and (accumulative_hist[j] >= QUANTILE):
        #         selected[0] = True
        #         in_range[0] = (j - h_offset) / h_rscale
        #     if (not selected[1]) and (accumulative_hist[j] >= 1 - QUANTILE):
        #         selected[1] = True
        #         in_range[1] = (j - h_offset) / h_rscale
        #
        in_scale = (in_range[1] - in_range[0]) / in_bins
        if in_step > 1e-9:
            in_scale = np.round(in_scale / in_step) * in_step
        in_offset = np.int32(np.round(-in_range[0] / in_scale))
        # quantizing weights
        if symetric_mode:
            if w_bins % 2 != 1:
                raise RuntimeError('Symetric requiers odd ammoint of w_bins')
            a = torch.quantile(torch.abs(self.f32_weights.data), 1 - QUANTILE).item()
            w_scale = 2 * a / w_bins
            w_offset = w_bins // 2
        else:
            r = torch.quantile(self.f32_weights.data, 1 - QUANTILE).item()
            l = torch.quantile(self.f32_weights.data, QUANTILE).item()
            w_scale = (r - l) / in_bins
            w_offset = np.int32(np.round(-l / w_scale))

        # finetuning weights
        device = self.f32_weights.device
        NREP = 50
        wsh = torch.normal(0, 0.02 * self.f32_weights.data.std().item(),
                           size=self.f32_weights.shape, device=device, requires_grad=True)
        bsh = torch.normal(0, 0.02 * self.f32_bias.data.std().item(),
                           size=self.f32_bias.shape, device=device, requires_grad=True)
        optimizer = optim.Adam([wsh, bsh], 1e-5)
        mse = nn.MSELoss()
        for cnt in range(NREP * len(cached_in)):
            optimizer.zero_grad()
            j = cnt % len(cached_in)
            x = fake_quant.apply(cached_in[j], in_scale, in_offset, 0, in_bins - 1)
            w = fake_quant.apply(wsh + self.f32_weights, w_scale, w_offset, 0, w_bins - 1)
            s = w_scale * in_scale
            b = scale_quant.apply(bsh + self.f32_bias, s) * s
            y = F.conv2d(x, w, b, stride=self.stride, padding=self.padding, groups=self.groups)
            loss = mse(y, cached_out[j])
            loss.backward()
            optimizer.step()

        self.f32_weights.data = self.f32_weights.data + wsh
        self.f32_bias.data = self.f32_bias.data + bsh
        self.input_quant = LinearQuant(in_bins, in_scale, in_offset)
        self.weight_quant = LinearQuant(w_bins, w_scale, w_offset)

    def forward(self, x):
        x = self.input_quant(x) - self.input_quant.offset
        w = self.weight_quant(self.f32_weights) - self.weight_quant.offset
        s = self.input_quant.scale * self.weight_quant.scale
        b = scale_quant.apply(self.f32_bias, s)
        y = F.conv2d(x, w, b, stride=self.stride, padding=self.padding, groups=self.groups)
        return y * s

    def cache(self, x):
        return self.forward(x)  # does not have cache

    def drop_cache(self):
        pass  # does not have cache


class FrozenConv2d(nn.Module):
    def __init__(self, freezing_layer):
        super().__init__()
        self.preprocess = None
        self.postprocess = None
        if isinstance(freezing_layer, FConv2d):
            self.in_channels = freezing_layer.conv.in_channels
            self.out_channels = freezing_layer.conv.out_channels
            self.padding = freezing_layer.conv.padding
            self.stride = freezing_layer.conv.stride
            self.groups = freezing_layer.conv.groups
            device = freezing_layer.conv.weight.device
            bn_scale = torch.ones((self.out_channels,), dtype=torch.float32, device=device)
            bias = torch.zeros((self.out_channels,), dtype=torch.float32, device=device)
            if freezing_layer.conv.bias is not None:
                bias = freezing_layer.conv.bias.data.clone()
            if freezing_layer.bn is not None:
                bn = freezing_layer.bn
                bn_scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
                bias = (bias - bn.running_mean) * bn_scale + bn.bias
            self.register_buffer('weight', freezing_layer.conv.weight.data.detach() * (
                bn_scale.view((self.out_channels, 1, 1, 1))).detach().detach())
            self.register_buffer('bias', bias.detach())
        elif isinstance(freezing_layer, QConv2d):
            self.in_channels = freezing_layer.in_channels
            self.out_channels = freezing_layer.out_channels
            self.padding = freezing_layer.padding
            self.stride = freezing_layer.stride
            self.groups = freezing_layer.groups
            lq = freezing_layer.input_quant
            wq = freezing_layer.weight_quant
            self.preprocess = LinearQuantShifted(lq.bins.item(), lq.scale.item(), lq.offset.item())
            self.register_buffer('weight_scale', torch.tensor([wq.scale.item()], dtype=torch.float32))
            self.register_buffer('weight_offset', torch.tensor([wq.offset.item()], dtype=torch.int32))
            s = lq.scale.item() * wq.scale.item()
            self.postprocess = Dequant(s)
            self.register_buffer('weight', torch.round(freezing_layer.f32_weights / wq.scale.item()).int().detach())
            self.register_buffer('bias', torch.round(freezing_layer.f32_bias / s).int().detach())
        else:
            raise RuntimeError('incorrect layer type')

    def get_weight(self):
        return self.weight

    def forward(self, x):
        if self.preprocess is not None:
            x = self.preprocess(x)
        x = F.conv2d(x, self.weight.float(), self.bias.float(),
                     stride=self.stride, padding=self.padding, groups=self.groups)
        if self.postprocess is not None:
            x = self.postprocess(x)
        return x

    def cache(self, x):
        return self.forward(x)  # does not have cache

    def drop_cache(self):
        pass  # does not have cache