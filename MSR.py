import torch
import torch.nn as nn
from backbone_utils import Backbone

class MSR(nn.Module):
    def __init__(self, layers, num_classes=10): 
        super(MSR, self).__init__()
        self.backbone = Backbone(
            'resnet{}'.format(layers), 
            train_backbone=False, 
            return_interm_layers=True, 
            dilation=[False, True, True]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)  # Assuming last feature map size is 2048
        self.num_classes = num_classes

    def forward(self, x):
        # Backbone feature extraction
        back_x = self.backbone(x)
        x = self.avgpool(back_x['3'])  # AdaptiveAvgPool the output of layer '3'
        x = torch.flatten(x, 1)  # Flatten the feature maps
        x = self.fc(x)  # Fully connected layer
        return x

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_fc_params(self):
        return self.fc.parameters()