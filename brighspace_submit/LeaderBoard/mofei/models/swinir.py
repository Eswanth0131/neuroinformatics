"""
SwinIR: Image Restoration Using Swin Transformer
Adapted for MRI restoration (low-field → high-field, already spatially aligned).

Reference: Liang et al., "SwinIR: Image Restoration Using Swin Transformer", ICCV 2021

This is the "image restoration" variant (no upsampling module) since
our input is already trilinear-upsampled to target resolution (179×221).
The network learns a residual: output = input + network(input).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import register


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


def window_partition(x, window_size):
    """Partition into non-overlapping windows.
    x: (B, H, W, C) → (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window_partition.
    windows: (num_windows*B, window_size, window_size, C) → (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention (W-MSA) with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (wh, ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, wh, ww)
        coords_flat = torch.flatten(coords, 1)  # (2, wh*ww)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (nH, N, N)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with optional shifted window."""

    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(window_size, window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, x_size, attn_mask):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # (nW*B, ws, ws, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA
        mask = attn_mask if self.shift_size > 0 else None
        attn_windows = self.attn(x_windows, mask=mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).
    N Swin Transformer layers + conv + residual connection.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
            )
            for i in range(depth)
        ])

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, x_size, attn_mask):
        shortcut = x
        for blk in self.blocks:
            x = blk(x, x_size, attn_mask)
        # Reshape for conv: (B, H*W, C) → (B, C, H, W)
        B, L, C = x.shape
        H, W = x_size
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        return x + shortcut


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

@register("swinir")
class SwinIR(nn.Module):
    """
    SwinIR for MRI restoration (no upsampling).
    Global residual: output = input + network(input).

    Default config (~1.4M params):
        embed_dim=90, depths=[4,4,4,4], num_heads=[6,6,6,6], window_size=8

    Input/Output: (B, 1, H, W)
    """

    def __init__(
        self,
        in_chans=1,
        embed_dim=90,
        depths=(4, 4, 4, 4),
        num_heads=(6, 6, 6, 6),
        window_size=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
    ):
        super().__init__()
        self.window_size = window_size
        num_layers = len(depths)

        # --- Shallow feature extraction ---
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # --- Deep feature extraction (RSTB stack) ---
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(RSTB(
                dim=embed_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            ))

        self.norm = nn.LayerNorm(embed_dim)

        # --- Reconstruction (image restoration: just a conv) ---
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

        # Precompute attention mask for known input size (179×221 → 184×224)
        # Registered as buffer so DataParallel replicates to each GPU
        self._input_h = 179
        self._input_w = 221
        Hp = self._input_h + (window_size - self._input_h % window_size) % window_size
        Wp = self._input_w + (window_size - self._input_w % window_size) % window_size
        self._padded_size = (Hp, Wp)
        self.register_buffer("attn_mask", self._make_attn_mask(Hp, Wp, window_size))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @staticmethod
    def _make_attn_mask(H, W, ws):
        """Compute attention mask for SW-MSA (shifted windows)."""
        shift = ws // 2
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        w_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, ws)
        mask_windows = mask_windows.view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        identity = x

        # Pad to multiple of window_size
        _, _, H, W = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, Hp, Wp = x.shape

        # Shallow features
        x = self.conv_first(x)
        shallow = x

        # Reshape for transformer: (B, C, H, W) → (B, H*W, C)
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, Hp * Wp, x.shape[1])
        x_size = (Hp, Wp)

        # Use precomputed mask (already on correct device via buffer)
        attn_mask = self.attn_mask

        # Deep features (RSTB stack)
        for layer in self.layers:
            x = layer(x, x_size, attn_mask)

        x = self.norm(x)

        # Reshape back: (B, H*W, C) → (B, C, H, W)
        B, L, C = x.shape
        x = x.view(B, Hp, Wp, C).permute(0, 3, 1, 2)

        # Reconstruction
        x = self.conv_after_body(x) + shallow
        x = self.conv_last(x)

        # Remove padding
        x = x[:, :, :H, :W]

        # Global residual
        return identity + x
