import torch
import torch.nn as nn


class MaskedConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation=1, groups=1, bias=True, causal=False):
        if causal:
            padding = (kernel_size - 1) * dilation
        else:
            padding = (kernel_size - 1) * dilation // 2
        super(MaskedConv1d, self).__init__(in_channels, out_channels, kernel_size,
                                           stride=1, padding=padding, dilation=dilation,
                                           groups=groups, bias=bias)

    def forward(self, inputs):
        output = super(MaskedConv1d, self).forward(inputs)
        return output[:, :, :inputs.size(2)]


class HighwayConv1d(MaskedConv1d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation=1, groups=1, bias=True, causal=False):
        super(HighwayConv1d, self).__init__(in_channels, 2 * out_channels, kernel_size,
                                            dilation, groups, bias, causal)
        self.sigmoid_ = nn.Sigmoid()

    def forward(self, inputs):
        L = super(HighwayConv1d, self).forward(inputs)
        H1, H2 = torch.chunk(L, 2, 1)  # chunk at the feature dim
        H1 = self.sigmoid_(H1)
        return H1 * H2 + (1.0 - H1) * inputs
