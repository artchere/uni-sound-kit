import torch
from torch import nn

from core.nn_modules.extractor import FeatureExtractor


class S1Block(nn.Module):
    """ S1 ConvBlock used in Temporal Convolutions
        -> CONV -> BN -> RELU -> CONV -> BN -> (+) -> RELU
        |_______________________________________|
    """
    def __init__(self, out_ch):
        super(S1Block, self).__init__()

        # First convolution layer
        self.conv0 = nn.Conv1d(in_channels=out_ch, out_channels=out_ch, kernel_size=9, stride=1,
                               padding=4, bias=False)
        self.bn0 = nn.BatchNorm1d(out_ch, affine=True)
        # Second convolution layer
        self.conv1 = nn.Conv1d(in_channels=out_ch, out_channels=out_ch, kernel_size=9, stride=1,
                               padding=4, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch, affine=True)

    def forward(self, x):
        identity = x

        x_out = self.conv0(x)
        x_out = self.bn0(x_out)
        x_out = nn.ReLU()(x_out)

        x_out = self.conv1(x_out)
        x_out = self.bn1(x_out)

        x_out += identity
        x_out = nn.ReLU()(x_out)

        return x_out


class S2Block(nn.Module):
    """ S2 ConvBlock used in Temporal Convolutions
        -> CONV -> BN -> RELU -> CONV -> BN -> (+) -> RELU
        |_______-> CONV -> BN -> RELU ->________|
    """
    def __init__(self, in_ch, out_ch):
        super(S2Block, self).__init__()

        # First convolution layer
        self.conv0 = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=9, stride=2,
                               padding=4, bias=False)
        self.bn0 = nn.BatchNorm1d(out_ch, affine=True)
        # Second convolution layer
        self.conv1 = nn.Conv1d(in_channels=out_ch, out_channels=out_ch, kernel_size=9, stride=1,
                               padding=4, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch, affine=True)
        # Residual convolution layer
        self.conv_res = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=9, stride=2,
                                  padding=4, bias=False)
        self.bn_res = nn.BatchNorm1d(out_ch, affine=True)

    def forward(self, x):
        x_out = self.conv0(x)
        x_out = self.bn0(x_out)
        x_out = nn.ReLU()(x_out)

        x_out = self.conv1(x_out)
        x_out = self.bn1(x_out)

        identity = self.conv_res(x)
        identity = self.bn_res(identity)
        identity = nn.ReLU()(identity)

        x_out += identity
        x_out = nn.ReLU()(x_out)

        return x_out


class TCResNet14(nn.Module):
    """
    Modified ori TC-ResNet14 arch with 1D-conv and lstm/attention layers as optional
    Tensor shape: [Batch x Channel (1) x Frequency x Time]
    """
    def __init__(
            self,
            multiplier: float = 1.0
    ):
        super(TCResNet14, self).__init__()

        # First Convolution layer
        self.conv_block = nn.LazyConv1d(
            out_channels=int(16 * multiplier),
            kernel_size=3,
            padding=1,
            bias=False
        )

        # Conv Blocks
        self.s2_block0 = S2Block(int(16 * multiplier), int(24 * multiplier))
        self.s1_block0 = S1Block(int(24 * multiplier))
        self.s2_block1 = S2Block(int(24 * multiplier), int(32 * multiplier))
        self.s1_block1 = S1Block(int(32 * multiplier))
        self.s2_block2 = S2Block(int(32 * multiplier), int(48 * multiplier))
        self.s1_block2 = S1Block(int(48 * multiplier))

    def forward(self, x):
        x = x.float()
        x_out = self.conv_block(x)
        x_out = self.s2_block0(x_out)
        x_out = self.s1_block0(x_out)
        x_out = self.s2_block1(x_out)
        x_out = self.s1_block1(x_out)
        x_out = self.s2_block2(x_out)
        x_out = self.s1_block2(x_out)

        return x_out


if __name__ == "__main__":
    extractor = FeatureExtractor(extractor_type="mfcc",
                                 use_delta_order=0,
                                 add_channel_dim=False,
                                 target_sr=16000,
                                 win_length=0.040,
                                 hop_length=0.010)

    # Test array like 1-channel audio with 5 sec length (batch size == 2):
    test_tensor = torch.randn(2, 1, 80000)
    # Get frequency/time sizes after feature extraction:
    freq_size, time_size = extractor(test_tensor).shape[-2:]

    print(extractor(test_tensor).shape)

    # Construct the final model:
    model = nn.Sequential(
        extractor,
        TCResNet14(multiplier=2)
    )

    print(model(test_tensor).shape)
