import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import PositionEmbeddingSine, Transformer

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, feature_dims):
        super(UNet, self).__init__()
        
        self.encoder1 = DoubleConv(in_channels, feature_dims)
        self.encoder2 = DoubleConv(feature_dims, feature_dims * 2)
        self.encoder3 = DoubleConv(feature_dims * 2, feature_dims * 4)
        self.encoder4 = DoubleConv(feature_dims * 4, feature_dims * 8)

        self.embed = nn.Embedding(feature_dims * 16,1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.pe_layer = PositionEmbeddingSine(feature_dims * 16 // 2, normalize=True)
        self.transformer = Transformer(
            d_model=feature_dims * 16,
            dropout=0.1,
            nhead=4,
            dim_feedforward=feature_dims * 4,
            num_encoder_layers=0,
            num_decoder_layers=1,
            normalize_before=False,
            return_intermediate_dec=False,
        )
        
        self.bottleneck_conv = DoubleConv(feature_dims * 8, feature_dims * 16)
        
        self.upconv4 = nn.ConvTranspose2d(feature_dims * 16, feature_dims * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(feature_dims * 16, feature_dims * 8)
        self.upconv3 = nn.ConvTranspose2d(feature_dims * 8, feature_dims * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(feature_dims * 8, feature_dims * 4)
        self.upconv2 = nn.ConvTranspose2d(feature_dims * 4, feature_dims * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(feature_dims * 4, feature_dims * 2)
        self.upconv1 = nn.ConvTranspose2d(feature_dims * 2, feature_dims, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(feature_dims * 2, feature_dims)
        
        self.final_conv = nn.Conv2d(feature_dims, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.pool(enc4)
        bottleneck = self.bottleneck_conv(bottleneck)
        
        b, c, h, w = bottleneck.shape
        bottleneck_avg = bottleneck.mean(dim=(2, 3), keepdim=True).squeeze(-1)
        bottleneck_pe = self.pe_layer(bottleneck)
        
        transformer_output = self.transformer(bottleneck, None, bottleneck_avg, self.embed.weight, bottleneck_pe) # (bsz,1024,1)

        transformer_output_expanded = transformer_output.unsqueeze(-1) # (bsz,1024,1,1)
        bottleneck = bottleneck * transformer_output_expanded
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.final_conv(dec1))
