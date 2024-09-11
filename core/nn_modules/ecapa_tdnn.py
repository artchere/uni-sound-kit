import torch
from torch import nn

from core.nn_modules.extractor import FeatureExtractor


class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        sp = None
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.convs[i](sp)
            sp = self.bns[i](nn.ReLU()(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out


class Conv1dReluBn(nn.Module):
    def __init__(self, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.LazyConv1d(out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(nn.ReLU()(self.conv(x)))


class SEConnect(nn.Module):
    def __init__(self, channels, s=2):
        super().__init__()
        assert channels % s == 0, "{} % {} != 0".format(channels, s)
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = nn.ReLU()(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out


def res2block(channels, kernel_size, stride, padding, dilation, scale):
    return nn.Sequential(
        Conv1dReluBn(out_channels=channels, kernel_size=1, stride=1, padding=0),
        Res2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale),
        Conv1dReluBn(out_channels=channels, kernel_size=1, stride=1, padding=0),
        SEConnect(channels)
    )


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        super().__init__()
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)  # equals V and k in the paper

    def forward(self, x):
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class ECAPATDNN(nn.Module):
    """
    Tensor shape: [Batch x Frequency x Time]
    """
    def __init__(self, channels=512, pre_emb_size=1536, post_emb_size=192):
        super(ECAPATDNN, self).__init__()
        self.layer1 = Conv1dReluBn(out_channels=channels, kernel_size=5, padding=2)
        self.layer2 = res2block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = res2block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = res2block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)

        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, pre_emb_size, kernel_size=1)
        self.pooling = AttentiveStatsPool(pre_emb_size, 128)
        self.bn1 = nn.BatchNorm1d(pre_emb_size*2)
        self.linear = nn.Linear(pre_emb_size*2, post_emb_size)
        self.bn2 = nn.BatchNorm1d(post_emb_size)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

        out = torch.cat([out2, out3, out4], dim=1)
        out = nn.ReLU()(self.conv(out))
        out = self.bn1(self.pooling(out))
        out = self.bn2(self.linear(out))

        return out


if __name__ == "__main__":
    extractor = FeatureExtractor(extractor_type="melspec",
                                 use_delta_order=0,
                                 add_channel_dim=False,
                                 target_sr=16000,
                                 win_length=0.040,
                                 hop_length=0.010)

    # Test array like 1-channel audio with 5 sec length (batch size == 2):
    test_tensor = torch.randn(2, 1, 64000)
    # Get frequency/time sizes after feature extraction:
    freq_size, time_size = extractor(test_tensor).shape[-2:]

    print(extractor(test_tensor).shape)

    # Construct the final model:
    model = nn.Sequential(
        extractor,
        ECAPATDNN(channels=8, pre_emb_size=1536, post_emb_size=192)
    )

    print(model(test_tensor).shape)
