import torch
from Backbone_44 import Backbone_44
from Channel_Attention import Channel_Attention
from Spatial_Attention import Spatial_Attention
from FC import FC

class Face_Fake_Net(torch.nn.Module):
    def __init__(self):
        super(Face_Fake_Net, self).__init__()
        self.backbone = Backbone_44()
        self.channel = Channel_Attention(channels=2048)
        self.spatial = Spatial_Attention()
        self.fc = FC(2097)

    def forward(self, x):
        x = self.backbone(x)
        y = self.channel(x)
        z = self.spatial(x)
        y = torch.flatten(y, start_dim=1)
        z = torch.flatten(z, start_dim=1)
        concatenated_tensor = torch.cat((y, z), dim=1)
        t = self.fc(concatenated_tensor)
        return t