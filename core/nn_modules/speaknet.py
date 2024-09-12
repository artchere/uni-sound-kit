import torch
from torch import nn

from core.nn_modules.extractor import FeatureExtractor


class SpeakNetResBlock(nn.Module):
    def __init__(self, filters=None, conv_num=None):
        super(SpeakNetResBlock, self).__init__()
        self.shortcut_conv = nn.LazyConv1d(out_channels=filters, kernel_size=1, padding='same')
        self.x_conv = nn.LazyConv1d(out_channels=filters, kernel_size=3, padding='same')
        self.convs = []
        for i in range(conv_num - 1):
            self.convs += [nn.LazyConv1d(out_channels=filters, kernel_size=3, padding='same'), nn.ReLU()]
        self.convs = nn.ModuleList(self.convs)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        s = self.shortcut_conv(x)
        for conv in self.convs:
            x = conv(x)
        x = self.x_conv(x)
        x += s
        x = self.relu(x)
        x = self.maxpool(x)

        return x


class SpeakNet(nn.Module):
    def __init__(self, n_filters=128):
        super(SpeakNet, self).__init__()
        self.resblocks = nn.Sequential(
            SpeakNetResBlock(n_filters//8, 2),
            SpeakNetResBlock(n_filters//4, 2),
            SpeakNetResBlock(n_filters//2, 3),
            SpeakNetResBlock(n_filters, 3),
            SpeakNetResBlock(n_filters, 3)
        )
        self.pool = nn.AvgPool1d(3, 3)

    def forward(self, x):
        x = self.resblocks(x)
        return self.pool(x)


if __name__ == "__main__":
    extractor = FeatureExtractor(extractor_type="mfcc",
                                 use_delta_order=2,
                                 add_channel_dim=False,
                                 target_sr=16000,
                                 win_length=0.040,
                                 hop_length=0.010)

    # Test array like 1-channel audio with 2 sec length (batch size == 2):
    test_tensor = torch.randn(2, 1, 32000)
    # Get frequency/time sizes after feature extraction:
    freq_size, time_size = extractor(test_tensor).shape[-2:]

    print(extractor(test_tensor).shape)

    # Construct the final model:
    model = nn.Sequential(
        extractor,
        SpeakNet(n_filters=128)
    )

    print(model(test_tensor).shape)
