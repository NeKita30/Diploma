import torch
import torch.nn as nn
from torch.autograd import Function


class fake_quant(Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, q_min, q_max):
        # q = torch.floor(x / scale + zero_point + 0.5).clip(q_min, q_max)
        # y = (q - zero_point) * scale
        y = torch.fake_quantize_per_tensor_affine(x, scale, zero_point, q_min, q_max)
        ctx.save_for_backward(x, y)
        ctx.scale = scale.item() if isinstance(scale, torch.Tensor) else scale
        ctx.zero_point = zero_point.item() if isinstance(zero_point, torch.Tensor) else zero_point
        ctx.q_min = q_min
        ctx.q_max = q_max

        if isinstance(zero_point, torch.Tensor):
            ctx.zero_point = zero_point.item()
        return y

    @staticmethod
    def backward(ctx, grad):
        grad_x = None
        grad_scale = None
        x, y = ctx.saved_tensors
        f_min = (ctx.q_min - ctx.zero_point) * ctx.scale
        f_max = (ctx.q_max - ctx.zero_point) * ctx.scale
        mask = (f_min <= x).to(grad.dtype).to(grad.device) * (x <= f_max).to(grad.dtype).to(grad.device)
        if ctx.needs_input_grad[0]:
            grad_x = grad * mask
        if ctx.needs_input_grad[1]:
            grad_scale = grad * y * (1 - mask) / ctx.scale

        return grad_x, grad_scale, None, None, None


class scale_quant(Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        x = x / scale + 0.5
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad):
        return grad / ctx.scale, None


class FakeLinearQuant(nn.Module):
    def __init__(self, bins, scale, offset):
        super().__init__()
        self.register_buffer('bins', torch.tensor([bins], dtype=torch.float32))
        self.register_buffer('scale', torch.tensor([scale], dtype=torch.float32))
        self.register_buffer('offset', torch.tensor([offset], dtype=torch.int32))

    def forward(self, x):
        y = fake_quant.apply(x, self.scale, self.offset, 0, self.bins - 1)
        return y


class LinearQuantShifted(nn.Module):
    def __init__(self, bins, scale, offset):
        super().__init__()
        self.register_buffer('bins', torch.tensor([bins], dtype=torch.int32))
        self.register_buffer('scale', torch.tensor([scale], dtype=torch.float32))
        self.register_buffer('offset', torch.tensor([offset], dtype=torch.int32))

    def forward(self, x):
        y = fake_quant.apply(x, self.scale, self.offset, 0, self.bins.item()) / self.scale
        return y


class LinearQuant(nn.Module):
    def __init__(self, bins, scale, offset):
        super().__init__()
        self.register_buffer('bins', torch.tensor([bins], dtype=torch.int32))
        self.register_buffer('scale', torch.tensor([scale], dtype=torch.float32))
        self.register_buffer('offset', torch.tensor([offset], dtype=torch.int32))

    def forward(self, x):
        y = fake_quant.apply(x, self.scale, self.offset, 0, self.bins.item()) / self.scale + self.offset
        return y


class Dequant(nn.Module):
    def __init__(self, out_scale):
        super().__init__()
        self.register_buffer('scale', torch.tensor([out_scale], dtype=torch.float32))

    def forward(self, x):
        return x * self.scale

