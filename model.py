
import torch.nn as nn


class TextureConversion(nn.Module):
    def __init__(self, channel = 1):
        super(TextureConversion, self).__init__()
        self.input_channel = channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channel, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 1, (3, 3), stride=(1, 1), padding=(1, 1))
        )

    def forward(self, x):
        x = self.conv(x)
        return x


