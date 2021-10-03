import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import math

# AttentionConv(planes, planes, kernel_size=7, padding=3, groups=8)
class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):

        # torch.Size([4, 64, 112, 112])
        batch, channels, height, width = x.size()
        # print(batch, channels, height, width )
        # 扩充x维度, 0补充    torch.Size([4, 64, 118, 118])
        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)
        # k_out torch.Size([4, 64, 118, 118])
        # print("q_out",q_out.size())
        # unfold先2后3(先对列取size=7 step=1, 再对行取size=7 step=1)
        # k_out为size = 7x7 step = 1 的滑动窗口提取的块
        # k_out.size() torch.Size([4, 64, 112, 112, 7, 7])
        # 每个块的大小为7x7, 每个通道上共112x112个块(118-7+1 = 112)
        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        # print("k_out",k_out.size())
        # print("v_out",v_out.size())
        # 按通道分成k_out_h和k_out_w
        # 对k_out_h加上二维相对列偏移, 对k_out_w加上二维相对行偏移
        # 恢复k_out的维度
        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        # print("k_out_h",k_out_h.size())
        # print("rel_h",self.rel_h.size())

        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)
        # print("k_out",k_out.size())

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        # print("k_out2", k_out.size())
        # print("v_out2", v_out.size())
        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)
        # print("q_out2",q_out.size())
        # q*(k+r)
        out = q_out * k_out

        out = F.softmax(out, dim=-1)
        # print("out2",out.size())

        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)
        # print("out4",out.size())
        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


class AttentionStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, m=4, bias=False):
        super(AttentionStem, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.m = m

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.emb_a = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_b = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_mix = nn.Parameter(torch.randn(m, out_channels // groups), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias) for _ in range(m)])

        self.reset_parameters()

    def forward(self, x):
        # x.Size([2, 3, 32, 32])
        batch, channels, height, width = x.size()
        # batch, channels, height, width 2 3 32 32

        # 上下左右各填充padding=2
        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        # padded_x.Size([2, 3, 36, 36])

        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = torch.stack([self.value_conv[_](padded_x) for _ in range(self.m)], dim=0)
        # q_out.Size([2, 64, 32, 32])
        # k_out.Size([2, 64, 36, 36])
        # v_out.Size([4, 2, 64, 36, 36])
        # k_out1 = k_out.unfold(2, self.kernel_size, self.stride)
        # v_out1 = v_out.unfold(3, self.kernel_size, self.stride)
        # print("k_out1", k_out1.size())
        # print("v_out1", v_out1.size())
        # # k_out1.Size([2, 64, 36, 33, 4])
        # # v_out1.Size([4, 2, 64, 36, 33, 4])
        #
        # k_out2 = k_out1.unfold(3, self.kernel_size, self.stride)
        # v_out2 = v_out1.unfold(4, self.kernel_size, self.stride)
        # # k_out2.Size([2, 64, 33, 33, 4, 4])
        # # v_out2.Size([4, 2, 64, 33, 33, 4, 4])
        # print("k_out2", k_out2.size())
        # print("v_out2", v_out2.size())
        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)
        # k_out.Size([2, 64, 33, 33, 4, 4])
        # v_out.Size([4, 2, 64, 33, 33, 4, 4])
        # if k_out == k_out2:


        k_out = k_out[:, :, :height, :width, :, :]
        v_out = v_out[:, :, :, :height, :width, :, :]

        emb_logit_a = torch.einsum('mc,ca->ma', self.emb_mix, self.emb_a)   # 行编码
        emb_logit_b = torch.einsum('mc,cb->mb', self.emb_mix, self.emb_b)   # 列编码
        emb = emb_logit_a.unsqueeze(2) + emb_logit_b.unsqueeze(1)
        emb = F.softmax(emb.view(self.m, -1), dim=0).view(self.m, 1, 1, 1, 1, self.kernel_size, self.kernel_size)

        v_out = emb * v_out

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(self.m, batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = torch.sum(v_out, dim=0).view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk->bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        for _ in self.value_conv:
            init.kaiming_normal_(_.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.emb_a, 0, 1)
        init.normal_(self.emb_b, 0, 1)
        init.normal_(self.emb_mix, 0, 1)


# temp = torch.randn((2, 3, 32, 32))
# # print(temp)
# # conv = AttentionConv(3, 16, kernel_size=3, padding=1)
# conv = AttentionStem(in_channels=3, out_channels=64, kernel_size=4, stride=1, padding=2, groups=1)
# print(conv(temp).size())
if __name__ == '__main__':

    temp = torch.randn((4, 64, 112, 112))

    model = AttentionConv(64, 64, kernel_size=5, padding=2, groups=8)   # 64, 128, 256, 512
    out = model.forward(temp)
    print()