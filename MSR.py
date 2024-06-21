import torch
import torch.nn as nn
from backbone_utils import Backbone
from transformer import PositionEmbeddingSine
from transformer import Transformer

class MSR(nn.Module):
    def __init__(self, layers, num_classes=2, reduce_dim=64): 
        super(MSR, self).__init__()
        self.backbone = Backbone(
            'resnet{}'.format(layers), 
            train_backbone=False, 
            return_interm_layers=True, 
            dilation=[False, True, True]
        )
        self.embed=nn.Embedding(reduce_dim,1) 
        self.pe_layer_1=PositionEmbeddingSine(reduce_dim//2, normalize=True)
        self.pe_layer_2=PositionEmbeddingSine(reduce_dim//2, normalize=True)
        self.pe_layer_3=PositionEmbeddingSine(reduce_dim//2, normalize=True)
        self.transformer_1 = Transformer(
            d_model=reduce_dim,
            dropout=0.1,
            nhead=4,
            dim_feedforward=reduce_dim//4,
            num_encoder_layers=0,
            num_decoder_layers=1,
            normalize_before=False,
            return_intermediate_dec=False,
        )

        self.transformer_2 = Transformer(
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
            nn.Conv2d(512, reduce_dim, kernel_size=1, padding=0, bias=False)                         
        )
        self.conv_red_2 = nn.Sequential(
            nn.Conv2d(1024, reduce_dim, kernel_size=1, padding=0, bias=False)                         
        )
        self.conv_red_3 = nn.Sequential(
            nn.Conv2d(2048, reduce_dim, kernel_size=1, padding=0, bias=False)                         
        )


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(reduce_dim*reduce_dim*3, num_classes)  # Assuming last feature map size
        self.num_classes = num_classes

    def forward(self, x):
        # Backbone feature extraction
        back_x = self.backbone(x)
        #x = self.avgpool(back_x['3'])  # AdaptiveAvgPool the output of layer '3'
        #print(back_x['1'].shape)
        #print(back_x['2'].shape)
        #print(back_x['3'].shape)
        #print(x.shape)

        red_back_1 = self.conv_red_1(back_x['1'])
        avg_back_1 = self.avgpool(red_back_1)

        red_back_2 = self.conv_red_2(back_x['2'])
        avg_back_2 = self.avgpool(red_back_2)

        red_back_3 = self.conv_red_3(back_x['3'])
        avg_back_3 = self.avgpool(red_back_3)


        masking=None
        query_pos = self.embed.weight

        key_embed = red_back_1
        query_embed = avg_back_1.squeeze(-1)
        key_pos = self.pe_layer_1(red_back_1)
        fg_embed_1=self.transformer_1(key_embed,masking,query_embed,query_pos,key_pos)


        key_embed = red_back_2
        query_embed = avg_back_2.squeeze(-1)
        key_pos = self.pe_layer_2(red_back_2)
        fg_embed_2=self.transformer_2(key_embed,masking,query_embed,query_pos,key_pos)

        
        key_embed = red_back_3
        query_embed = avg_back_3.squeeze(-1)
        key_pos = self.pe_layer_3(red_back_3)
        fg_embed_3=self.transformer_3(key_embed,masking,query_embed,query_pos,key_pos)



        out_back_1=torch.sigmoid(torch.einsum("bchw,bcl->blhw",red_back_1,fg_embed_1)).permute(0, 2, 3, 1)
        out_back_2=torch.sigmoid(torch.einsum("bchw,bcl->blhw",red_back_2,fg_embed_2)).permute(0, 2, 3, 1)
        out_back_3=torch.sigmoid(torch.einsum("bchw,bcl->blhw",red_back_3,fg_embed_3)).permute(0, 2, 3, 1)

        # Concatenate along the last dimension
        out = torch.cat((out_back_1, out_back_2, out_back_3), dim=-1)


        out_1 = torch.flatten(out, 1)  # Flatten the feature maps
        out_1 = self.fc(out_1)  # Fully connected layer
        return out_1

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_fc_params(self):
        return self.fc.parameters()
