import torch
import torch.nn as nn

# Double Convolution Block used in U-Net
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        # Two convolution layers followed by ReLU activation
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder (Downsampling path)
        self.down1 = DoubleConv(3, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder (Upsampling path)
        self.up_trans1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = DoubleConv(256, 128)

        self.up_trans2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2 = DoubleConv(128, 64)

        # Final output layer (1 channel mask)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))

        # Decoder
        u1 = self.up_trans1(d3)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.up1(u1)

        u2 = self.up_trans2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.up2(u2)

        return torch.sigmoid(self.final(u2))
