import torch
import torch.nn as nn
import torch.nn.functional as F


def padding(batch, seq_len):
    if len(batch[0][0]) < seq_len:
        m = torch.nn.ConstantPad1d((0, seq_len - len(batch[0][0])), 0)
        batch = m(batch)
    return batch


class TCSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(TCSConv, self).__init__()

        self.depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size, groups=in_channels,
                                        padding='same')  # effectively performing a depthwise convolution
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels,
                                        kernel_size=1)  # effectively performing a pointwise convolution

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        return x


class SubBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SubBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.tcs_conv = TCSConv(self.in_channels, self.out_channels, self.kernel_size)
        self.bnorm = nn.BatchNorm1d(self.out_channels)
        self.dropout = nn.Dropout()

    def forward(self, x, residual=None):
        x = self.tcs_conv(x)
        x = self.bnorm(x)

        # apply the residual if passed
        if residual is not None:
            x = x + residual

        x = F.relu(x)
        x = self.dropout(x)

        return x


class MainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, repeat_blocks=1):
        super(MainBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.residual_pointwise = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1)
        self.residual_batchnorm = nn.BatchNorm1d(self.out_channels)

        self.sub_blocks = nn.ModuleList()

        self.sub_blocks.append(
            SubBlock(self.in_channels, self.out_channels, self.kernel_size)
        )

        for i in range(repeat_blocks - 1):
            self.sub_blocks.append(
                SubBlock(self.out_channels, self.out_channels, self.kernel_size)
            )

    def forward(self, x):
        residual = self.residual_pointwise(x)
        residual = self.residual_batchnorm(residual)

        for i, layer in enumerate(self.sub_blocks):
            if (i + 1) == len(self.sub_blocks):
                x = layer(x, residual)
            else:
                x = layer(x)

        return x


class MatchboxNet(nn.Module):
    """
    :param res_blocks: The number of residual blocks in the model
    :param sub_blocks: The number of sub-blocks within each residual block
    :param out_channels: The size of the output channels within a sub-block
    """
    def __init__(self, res_blocks=3, sub_blocks=2, out_channels=64, freq_size=64, kernel_sizes=None, out_dim=30):
        super(MatchboxNet, self).__init__()
        if not kernel_sizes:
            kernel_sizes = [k * 2 + 11 for k in range(1, 5 + 1)]

        # the prologue layers
        self.prologue_conv1 = nn.Conv1d(freq_size, 128, kernel_size=11, stride=2)
        self.prologue_bnorm1 = nn.BatchNorm1d(128)

        # the intermediate blocks
        self.blocks = nn.ModuleList()

        self.blocks.append(
            MainBlock(128, out_channels, kernel_sizes[0], repeat_blocks=sub_blocks)
        )

        for i in range(1, res_blocks):
            self.blocks.append(
                MainBlock(out_channels, out_channels, kernel_size=kernel_sizes[i], repeat_blocks=sub_blocks)
            )

        # the epilogue layers
        self.epilogue_conv1 = nn.Conv1d(out_channels, 128, kernel_size=29, dilation=2)
        self.epilogue_bnorm1 = nn.BatchNorm1d(128)

        self.epilogue_conv2 = nn.Conv1d(128, 128, kernel_size=1)
        self.epilogue_bnorm2 = nn.BatchNorm1d(128)

        self.epilogue_conv3 = nn.Conv1d(128, out_dim, kernel_size=1)

        # Pool the timesteps into a single dimension using simple average pooling
        self.epilogue_adaptivepool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = padding(x, 128)
        # prologue block
        x = self.prologue_conv1(x)
        x = self.prologue_bnorm1(x)
        x = F.relu(x)

        # intermediate blocks
        for layer in self.blocks:
            x = layer(x)

        # epilogue blocks
        x = self.epilogue_conv1(x)
        x = self.epilogue_bnorm1(x)

        x = self.epilogue_conv2(x)
        x = self.epilogue_bnorm2(x)

        x = self.epilogue_conv3(x)
        # x = self.epilogue_adaptivepool(x)
        # x = x.squeeze(2)  # (N, 30, 1) > (N, 30)
        # x = F.softmax(x, dim=1)  # softmax across classes and not batch

        return x


if __name__ == "__main__":
    model = MatchboxNet(3, 2, 64, 40, None, 16)

    print(model(torch.randn(2, 40, 375)).shape)
