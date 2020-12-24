import torch.nn as nn
import torch


class one_conv(nn.Module):
    def __init__(self, inchanels, growth_rate, kernel_size=3):
        super(one_conv, self).__init__()
        self.conv = nn.Conv2d(
            inchanels, growth_rate, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), 1)


class RDB(nn.Module):
    def __init__(self, G0, C, G, kernel_size=3):
        super(RDB, self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+i*G, G))
        self.conv = nn.Sequential(*convs)
        # local_feature_fusion
        self.LFF = nn.Conv2d(G0+C*G, G0, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        # local residual learning
        return lff + x


class RDNBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(RDNBlock, self).__init__()
        '''
        D: RDB number 20
        C: the number of conv layer in RDB 6
        G: the growth rate 32
        G0:local and global feature fusion layers 64 filter
        '''
        self.D = 3
        self.C = 3
        self.G = gc
        self.G0 = 64
        kernel_size = 3
        input_channels = channel_in
        # shallow feature extraction
        self.SFE1 = nn.Conv2d(
            input_channels, self.G0, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
        self.SFE2 = nn.Conv2d(
            self.G0, self.G0, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
        # RDB for paper we have D RDB block
        self.RDBS = nn.ModuleList()
        for d in range(self.D):
            self.RDBS.append(RDB(self.G0, self.C, self.G, kernel_size))
        # Global feature fusion
        self.GFF = nn.Sequential(
            nn.Conv2d(self.D*self.G0, self.G0,
                      kernel_size=1, padding=0, stride=1),
            nn.Conv2d(self.G0, self.G0, kernel_size,
                      padding=kernel_size//2, stride=1),
        )
        # upsample net
        self.up_net = nn.Sequential(
            nn.Conv2d(self.G0, self.G*4, kernel_size=kernel_size,
                      padding=kernel_size//2, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(self.G, self.G*4, kernel_size=kernel_size,
                      padding=kernel_size//2, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(self.G, channel_out, kernel_size=kernel_size,
                      padding=kernel_size//2, stride=1)
        )
        # init
        for para in self.modules():
            if isinstance(para, nn.Conv2d):
                nn.init.kaiming_normal_(para.weight)
                if para.bias is not None:
                    para.bias.data.zero_()

    def forward(self, x):
        # f-1
        F_h = self.SFE1(x)
        out = self.SFE2(F_h)
        RDB_outs = []
        for i in range(self.D):
            out = self.RDBS[i](out)
            RDB_outs.append(out)
        out = torch.cat(RDB_outs, 1)
        out = self.GFF(out)
        out = F_h + out
        return self.up_net(out)
