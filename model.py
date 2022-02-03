
import torch.nn as nn

model_tc = nn.Sequential(
    nn.Conv2d(1, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(),
    nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(),
    nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(),
    nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(),
    nn.Conv2d(64, 1, (3, 3), stride=(1, 1), padding=(1, 1))
)


class TextureConversion(nn.Module):
    def __init__(self):
        super(TextureConversion, self).__init__()
        self.conv = model_tc

    def forward(self, x):
        x = self.conv(x)
        return x


