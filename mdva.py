import torch
import torch.nn as nn
from torch import Tensor
from utils import ConvBNAct, DR_Pool

class AttentionNode(nn.Module):
    def __init__(self):
        super(AttentionNode, self).__init__()
        kernel_size = 7
        self.pool = DR_Pool()
        self.conv = ConvBNAct(2, 1, kernel_size)
    def forward(self, x: Tensor) -> Tensor:
        x_pool = self.pool(x)
        x_out = self.conv(x_pool)
        scale = torch.sigmoid(x_out)
        return scale


class MambaLikeAttention(nn.Module):
    def __init__(self, in_dim, no_spatial=False):
        super(MambaLikeAttention, self).__init__()
        self.cw = AttentionNode()
        self.ch = AttentionNode()
        self.no_spatial = no_spatial
        self.norm = nn.BatchNorm2d(in_dim)
        self.act = nn.ReLU(inplace=True)
        if not no_spatial:
            self.hw = AttentionNode()

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.norm(x)

        left_x = self.act(x)

        a = x
        b = x.permute(0, 2, 1, 3).contiguous()
        c = x.permute(0, 3, 2, 1).contiguous()

        cw_out = self.cw(b).permute(0, 2, 1, 3).contiguous()
        ch_out = self.ch(c).permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            hw_out = self.hw(a)
            out = 1/3 * (hw_out + cw_out + ch_out) *  left_x
        else:
            out = 1/2 * (cw_out + ch_out)  * left_x
        return out + identity





