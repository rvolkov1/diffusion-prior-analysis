import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb).to(x.device)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, class_emb_dim=None):
        super().__init__()
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        if class_emb_dim==None:
            class_emb_dim=time_emb_dim

        self.class_emb_proj = nn.Identity() if class_emb_dim is None else nn.Sequential(
            nn.SiLU(),
            nn.Linear(class_emb_dim, out_channels)
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb, c_emb=None):
        h = self.block1(x)
        t = self.time_emb_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        t = t.repeat(1, 1, h.shape[2], h.shape[3])  # Repeat along spatial dimensions
        c = 0 if c_emb is None else self.class_emb_proj(c_emb).unsqueeze(-1).unsqueeze(-1)
        c = c.repeat(1, 1, h.shape[2], h.shape[3])  # Repeat across spatial dimensions
        h = h + t + c
        h = self.block2(h)
        return h + self.residual_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).reshape(B, C, H * W).permute(0, 2, 1)
        k = self.k(h).reshape(B, C, H * W)
        v = self.v(h).reshape(B, C, H * W).permute(0, 2, 1)

        attn = torch.bmm(q, k) / math.sqrt(C)
        attn = attn.softmax(dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1).reshape(B, C, H, W)
        return self.proj(out) + x

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
    
class MiddleBlock(nn.Module):
    def __init__(self, channels, time_emb_dim):
        super().__init__()
        self.res_block = ResidualBlock(channels, channels, time_emb_dim, class_emb_dim=time_emb_dim)
        self.attn = AttentionBlock(channels)

    def forward(self, x, t_emb, c_emb):
        x = self.res_block(x, t_emb, c_emb)
        x = self.attn(x)
        return x


class UNet2(nn.Module):
    def __init__(self, in_channels=3, num_classes=100, base_channels=128, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.class_emb = nn.Embedding(num_classes, time_emb_dim)

        # Downsampling path
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = ResidualBlock(base_channels, base_channels, time_emb_dim)
        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down3 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.pool1 = Downsample(base_channels)
        self.pool2 = Downsample(base_channels * 2)

        # Bottleneck
        self.middle = MiddleBlock(base_channels * 4, time_emb_dim)

        # self.middle = nn.Sequential(
        #     ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim),
        #     AttentionBlock(base_channels * 4),
        #     ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim),
        # )

        # Upsampling path
        self.up3 = ResidualBlock(base_channels * 4 + base_channels * 2, base_channels * 2, time_emb_dim)
        self.up2 = ResidualBlock(base_channels * 2 + base_channels, base_channels, time_emb_dim)
        self.up1 = ResidualBlock(base_channels + 3, base_channels, time_emb_dim)

        self.up_sample2 = Upsample(base_channels * 4)
        self.up_sample1 = Upsample(base_channels * 2)

        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, 3, kernel_size=1)
        )

    def forward(self, x, y, t):
        t_emb = self.time_mlp(t)
        y_emb = self.class_emb(y)

        x1 = self.init_conv(x)
        x2 = self.down1(x1, t_emb, y_emb)
        x3 = self.pool1(x2)
        x4 = self.down2(x3, t_emb, y_emb)
        x5 = self.pool2(x4)
        x6 = self.down3(x5, t_emb, y_emb)

        mid = self.middle(x6, t_emb, y_emb)

        u3 = self.up_sample2(mid)
        u3 = self.up3(torch.cat([u3, x4], dim=1), t_emb, y_emb)

        u2 = self.up_sample1(u3)
        u2 = self.up2(torch.cat([u2, x2], dim=1), t_emb, y_emb)

        u1 = self.up1(torch.cat([u2, x], dim=1), t_emb, y_emb)

        return self.final_conv(u1)
