import torch
import torch.nn as nn
import torch.nn.functional as F

class CondMLP(nn.Module):
    def __init__(self):
        super(CondMLP, self).__init__()

        self.l0 = nn.Conv1d(84, 256, 1)
        self.res1 = ResBlock()
        self.res2 = ResBlock()
        self.res3 = ResBlock()
        self.res4 = ResBlock()
        self.res5 = ResBlock()

    def forward(self, x, z):

        x = self.l0(x)
        x = self.res1(x, z)
        x = self.res2(x, z)
        x = self.res3(x, z)
        x = self.res4(x, z)
        x = self.res5(x, z)

        return x


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.lz = nn.Conv1d(256, 256, 1)
        self.l1 = nn.Conv1d(256, 256, 1)
        self.l2 = nn.Conv1d(256, 256, 1)

    def forward(self, x, z):
        z = F.relu(self.lz(z))
        res = x + z
        x = F.relu(self.l1(res))
        x = F.relu(self.l2(x)) + res
        return x


class NeRF(nn.Module):
    def __init__(self):
        super(NeRF, self).__init__()
        self.mlp = CondMLP()
        self.alpha_l1 = nn.Conv1d(256, 1, 1)
        self.rgb_l1 = nn.Conv1d(310, 128, 1) # for shallow net
        self.rgb_l2 = nn.Conv1d(128, 3, 1)
        
    def query_sigma(x,z, masks = None, return_feat = True):
        feat = self.mlp(x[masks], z)
        # density
        alpha = self.alpha_l1(feat)
        
        if return_feat:
            return alpha, feat
        else:
            return alpha

    def forward(self, x, d, z):
        feat = self.mlp(x, z)
        # density
        alpha = self.alpha_l1(feat)
        # rgb
        feat = torch.cat((feat, d), dim=1)
        rgb = F.relu(self.rgb_l1(feat))
        rgb = self.rgb_l2(rgb)
        return rgb, alpha