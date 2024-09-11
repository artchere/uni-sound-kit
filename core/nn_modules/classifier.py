import torch
from torch import nn


# TODO: freq or time context size?


class LinearHead(nn.Module):
    """
    Classification head with Softmax out.
    Tensor shape: [Batch x Channel (1) x Frequency x Time], [Batch x Frequency x Time] or [Batch x Time]
    """
    def __init__(
            self,
            dropout: float = 0.25,
            hidden_units: list = None,
            n_classes: int = None,
            norm_layer: bool = True
    ):
        super(LinearHead, self).__init__()
        self.linear_projection = nn.Sequential(
            nn.LayerNorm(hidden_units),
            nn.Flatten(),
            nn.GELU(),
            nn.LazyLinear(max(hidden_units)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LazyLinear(n_classes),
            nn.Softmax(1)
        ) if norm_layer else nn.Sequential(
            nn.Flatten(),
            nn.GELU(),
            nn.LazyLinear(max(hidden_units)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LazyLinear(n_classes),
            nn.Softmax(1)
        )

    def forward(self, x):
        assert 2 <= len(x.shape) <= 4, "Incompatible tensor shape"

        if len(x.shape) == 4:
            assert x.shape[1] == 1, "The 1-st dim of the tensor must be the single-channel dim"
            x = x.squeeze(1)

        x = x.float()
        x = self.linear_projection(x)

        return x


class LSTMHead(nn.Module):
    """
    Classification head with Softmax out.
    Tensor shape: [Batch x Channel (1) x Frequency x Time], [Batch x Frequency x Time] or [Batch x Time]
    """
    def __init__(
            self,
            size: int = None,
            n_classes: int = None
    ):
        super(LSTMHead, self).__init__()
        self.n_layers = 1
        self.hidden_units = 192
        self.n_classes = n_classes
        self.linear_projection = nn.Sequential(
                nn.Flatten(),
                nn.ReLU(),
                nn.LazyLinear(self.n_classes),
                nn.Softmax(1)
            )
        self.lstm = nn.LSTM(
            input_size=size,
            hidden_size=self.hidden_units,
            batch_first=True,
            num_layers=self.n_layers
        )
        self.attn = TemporalAttn(size=self.hidden_units)

    def forward(self, x):
        assert 2 <= len(x.shape) <= 4, "Incompatible tensor shape"

        if len(x.shape) == 4:
            assert x.shape[1] == 1, "The 1-st dim of the tensor must be the same of 1"
            x = x.squeeze(1)
        elif len(x.shape) == 2:
            x = x.unsqueeze(-1)

        x = x.float()
        x = x.permute(0, -1, 1)  # [BxTxF]

        x, _ = self.lstm(x)  # ignore the hidden states (h0, c0)
        x = nn.Tanh()(x)
        x, weights = self.attn(x)

        x = self.linear_projection(x)

        return x


class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            padding=0,
            bias=False
        )

    def forward(self, x):
        return self.op(x)


class SpatialAttn(nn.Module):
    def __init__(self, in_features):
        super(SpatialAttn, self).__init__()
        self.op = nn.Conv2d(
            in_channels=in_features,
            out_channels=1,
            kernel_size=1,
            padding=0,
            bias=False
        )

    def forward(self, ll, g):
        batch, channel, units, time = ll.size()
        c = self.op(ll+g)
        a = nn.Softmax(dim=2)(c.view(batch, 1, -1)).view(batch, 1, units, time)
        g = torch.mul(a.expand_as(ll), ll)
        g = g.view(batch, channel, -1).sum(dim=2)

        return c.view(batch, 1, units, time), g


class TemporalAttn(nn.Module):
    def __init__(self, size):
        super(TemporalAttn, self).__init__()
        self.hidden_size = size
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)

    def forward(self, hidden_states):
        # [batch_size, time_steps, hidden_size]
        score_first_part = self.fc1(hidden_states)
        # [batch_size, hidden_size]
        h_t = hidden_states[:, -1, :]
        # [batch_size, time_steps]
        score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)
        attention_weights = nn.Softmax(dim=1)(score)
        # [batch_size, hidden_size]
        # context_vector = torch.bmm(hidden_states.permute(0,2,1), attention_weights.unsqueeze(2)).squeeze(2)\
        context_vector = torch.bmm(hidden_states.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2)
        # [batch_size, hidden_size*2]
        pre_activation = torch.cat((context_vector, h_t), dim=1)
        # [batch_size, hidden_size]
        attention_vector = self.fc2(pre_activation)
        attention_vector = nn.Tanh()(attention_vector)

        return attention_vector, attention_weights


if __name__ == '__main__':
    # spatial block
    spatial_block = SpatialAttn(in_features=3)
    l_test = torch.randn(16, 3, 128, 128)
    g_test = torch.randn(16, 3, 128, 128)
    result = spatial_block(l_test, g_test)
    print(result[0].shape)  # [Bx1xFxT]
    print(result[1].shape)  # [BxC]

    # temporal block
    temporal_block = TemporalAttn(size=256)
    permuted = torch.randn(4, 40, 256)  # [BxTxF]
    att_vector, att_weights = temporal_block(permuted)
    print(att_vector.shape)  # [BxF]
    print(att_weights.shape)  # [BxT]

    m = LSTMHead(size=40, n_classes=2)
    print(m(torch.randn(4, 40, 200)))
