import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone_utils import Backbone
from .transformer import PositionEmbeddingSine
from .transformer import Transformer

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return attention

class CRAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CRAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

class FeatureEnhancementBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureEnhancementBlock, self).__init__()
        self.cram = CRAM(in_channels)
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        ) 
        
    def forward(self, x):
        residual = self.residual(x)
        x = self.cram(x)
        return x + residual

class MSR(nn.Module):
    def __init__(self, layers, num_classes=2, reduce_dim=256): 
        super(MSR, self).__init__()
        self.backbone = Backbone(
            'resnet{}'.format(layers), 
            train_backbone=False, 
            return_interm_layers=True, 
            dilation=[False, True, True]
        )
        
        # Feature enhancement modules - FIXED CHANNEL DIMENSIONS
        self.enhancement_cat = FeatureEnhancementBlock(reduce_dim, reduce_dim)  # Changed from (512+1024, reduce_dim)
        self.enhancement_3 = FeatureEnhancementBlock(reduce_dim, reduce_dim)    # Changed from (2048, reduce_dim)
        
        self.embed_cat = nn.Embedding(reduce_dim, 1) 
        self.embed_3 = nn.Embedding(reduce_dim, 1) 
        self.pe_layer_cat = PositionEmbeddingSine(reduce_dim//2, normalize=True)
        self.pe_layer_3 = PositionEmbeddingSine(reduce_dim//2, normalize=True)

        self.transformer_cat = Transformer(
            d_model=reduce_dim,
            dropout=0.1,
            nhead=4,
            dim_feedforward=reduce_dim//4,
            num_encoder_layers=0,
            num_decoder_layers=1,
            normalize_before=False,
            return_intermediate_dec=False,
        )

        self.transformer_3 = Transformer(
            d_model=reduce_dim,
            dropout=0.1,
            nhead=4,
            dim_feedforward=reduce_dim//4,
            num_encoder_layers=0,
            num_decoder_layers=1,
            normalize_before=False,
            return_intermediate_dec=False,
        )
        
        # Convolution layers for feature reduction
        self.conv_red_1 = nn.Sequential(
            nn.Conv2d(512, reduce_dim//2, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(reduce_dim//2),
            nn.ReLU()
        )
        self.conv_red_2 = nn.Sequential(
            nn.Conv2d(1024, reduce_dim//2, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(reduce_dim//2),
            nn.ReLU()
        )
        self.conv_red_3 = nn.Sequential(
            nn.Conv2d(2048, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU()
        )

        # Cross-scale attention fusion
        self.cross_scale_attention = nn.MultiheadAttention(reduce_dim, 4, batch_first=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(reduce_dim * 2, reduce_dim),
            nn.BatchNorm1d(reduce_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(reduce_dim, reduce_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(reduce_dim // 2, num_classes)
        )
        
        self.num_classes = num_classes

    def forward(self, x):
        # Backbone feature extraction
        back_x = self.backbone(x)
        
        # Process individual feature maps
        red_back_1 = self.conv_red_1(back_x['1'])
        red_back_2 = self.conv_red_2(back_x['2'])
        red_back_3 = self.conv_red_3(back_x['3'])
        
        # Concatenate features (now both have reduce_dim channels)
        red_back_cat = torch.cat((red_back_1, red_back_2), dim=1)
        
        # Apply feature enhancement blocks
        enhanced_cat = self.enhancement_cat(red_back_cat)
        enhanced_3 = self.enhancement_3(red_back_3)
        
        # Prepare for transformer
        avg_back_cat = self.avgpool(enhanced_cat)
        avg_back_3 = self.avgpool(enhanced_3)

        masking = None
        query_pos = self.embed_cat.weight

        # Transformer processing
        key_embed = enhanced_cat
        query_embed = avg_back_cat.squeeze(-1)
        key_pos = self.pe_layer_cat(enhanced_cat)
        fg_embed_cat = self.transformer_cat(key_embed, masking, query_embed, query_pos, key_pos)
        
        query_pos = self.embed_3.weight
        key_embed = enhanced_3
        query_embed = avg_back_3.squeeze(-1)
        key_pos = self.pe_layer_3(enhanced_3)
        fg_embed_3 = self.transformer_3(key_embed, masking, query_embed, query_pos, key_pos)
        
        # Final concatenation and classification
        out = torch.cat((fg_embed_cat, fg_embed_3), dim=1)     
        out_1 = torch.flatten(out, 1)
        out_1 = self.classifier(out_1)
        
        return out_1

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_fc_params(self):
        return self.classifier.parameters()
