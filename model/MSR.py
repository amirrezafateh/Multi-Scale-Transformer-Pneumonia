import torch
import torch.nn as nn
from .backbone_utils import Backbone
from .transformer import PositionEmbeddingSine
from .transformer import Transformer

class MSR(nn.Module):
    def __init__(self, layers, num_classes=2, reduce_dim=256): 
        super(MSR, self).__init__()
        self.backbone = Backbone(
            'resnet{}'.format(layers), 
            train_backbone=False, 
            return_interm_layers=True, 
            dilation=[False, True, True]
        )
        self.embed_cat=nn.Embedding(reduce_dim,1) 
        self.embed_3=nn.Embedding(reduce_dim,1) 
        self.pe_layer_cat=PositionEmbeddingSine(reduce_dim//2, normalize=True)
        self.pe_layer_3=PositionEmbeddingSine(reduce_dim//2, normalize=True)

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
        self.conv_red_1 = nn.Sequential(
            nn.Conv2d(512, reduce_dim//2, kernel_size=1, padding=0, bias=False)                         
        )
        self.conv_red_2 = nn.Sequential(
            nn.Conv2d(1024, reduce_dim//2, kernel_size=1, padding=0, bias=False)                         
        )
        self.conv_red_3 = nn.Sequential(
            nn.Conv2d(2048, reduce_dim, kernel_size=1, padding=0, bias=False)                         
        )


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(reduce_dim*2, reduce_dim)  # Assuming last feature map size
        self.fc_2 = nn.Linear(reduce_dim, num_classes)  # Assuming last feature map size
        self.num_classes = num_classes

    def forward(self, x):
        # Backbone feature extraction
        back_x = self.backbone(x)
        red_back_1 = self.conv_red_1(back_x['1'])
        red_back_2 = self.conv_red_2(back_x['2'])
        
        red_back_cat = torch.cat((red_back_1, red_back_2), dim=1)
        avg_back_cat = self.avgpool(red_back_cat)
        
        red_back_3 = self.conv_red_3(back_x['3'])
        avg_back_3 = self.avgpool(red_back_3)

        masking=None
        query_pos = self.embed_cat.weight

        key_embed = red_back_cat
        query_embed = avg_back_cat.squeeze(-1)
        key_pos = self.pe_layer_cat(red_back_cat)

        fg_embed_cat=self.transformer_cat(key_embed,masking,query_embed,query_pos,key_pos)
        
        query_pos = self.embed_3.weight
        key_embed = red_back_3
        query_embed = avg_back_3.squeeze(-1)
        key_pos = self.pe_layer_3(red_back_3)
        fg_embed_3=self.transformer_3(key_embed,masking,query_embed,query_pos,key_pos)
 

        #out_back_cat = (torch.einsum("bchw,bcl->blhw",red_back_cat,fg_embed_cat)).permute(0, 2, 3, 1)
        #out_back_3 = (torch.einsum("bchw,bcl->blhw",red_back_3,fg_embed_3)).permute(0, 2, 3, 1)
        
        
        
        # Concatenate along the last dimension
        out = torch.cat((fg_embed_cat, fg_embed_3), dim=1)     

        out_1 = torch.flatten(out, 1)  # Flatten the feature maps
        
        out_1 = self.fc(out_1)  # Fully connected layer
        out_1 = self.fc_2(out_1)  # Fully connected layer
        return out_1

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_fc_params(self):
        return self.fc.parameters()
