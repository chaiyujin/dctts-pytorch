import torch
import torch.nn as nn


class MaskedConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation=1, padding="same", groups=1, bias=True):
        if padding == "causal":
            padding = (kernel_size - 1) * dilation
        elif padding == "same":
            padding = ((kernel_size - 1) * dilation + 1) // 2
        elif padding == "valid":
            padding = 0
        else:
            raise ValueError("[MaskedConv1d]: padding shoule be 'valid', 'same' or 'causal'")
        super(MaskedConv1d, self).__init__(in_channels, out_channels, kernel_size,
                                           stride=1, padding=padding, dilation=dilation,
                                           groups=groups, bias=bias)

    def forward(self, inputs):
        output = super(MaskedConv1d, self).forward(inputs)
        return output[:, :, :inputs.size(2)]


class HighwayConv1d(MaskedConv1d):
    def __init__(self, in_channels, kernel_size,
                 dilation=1, padding="same", groups=1, bias=True):
        super(HighwayConv1d, self).__init__(in_channels, 2 * in_channels, kernel_size,
                                            dilation, padding, groups, bias)
        self.sigmoid_ = nn.Sigmoid()

    def forward(self, inputs):
        L = super(HighwayConv1d, self).forward(inputs)
        H1, H2 = torch.chunk(L, 2, 1)  # chunk at the feature dim
        H1 = self.sigmoid_(H1)
        return H1 * H2 + (1.0 - H1) * inputs


class Deconv1d(nn.ConvTranspose1d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation=1, padding="same", groups=1, bias=True):
        if padding == "same":
            padding = max(0, (kernel_size - 2) // 2)
        super(Deconv1d, self).__init__(in_channels, out_channels, kernel_size,
                                       stride=2, padding=padding, groups=groups,
                                       bias=bias, dilation=dilation)
    def forward(self, inputs):
        return super(Deconv1d, self).forward(inputs)
