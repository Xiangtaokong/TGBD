from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
import math

from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

class BrainNetwork_mindeye(nn.Module):
    def __init__(self, out_dim=257*1024, in_dim=18000, clip_size=1024, h=4096, n_blocks=4, norm_type='ln', act_first=False, use_projector=True, subj='all'):
    # 15724 def __init__(self, out_dim=768, in_dim=18000, clip_size=768, h=4096, n_blocks=4, norm_type='ln', act_first=False, use_projector=True,dropout_p='ddd'):
        super().__init__()
        norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm, normalized_shape=h)
        act_fn = partial(nn.ReLU, inplace=True) if norm_type == 'bn' else nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)
        # self.temp = nn.Parameter(torch.tensor(.006))

        if subj[0] == 'subj01':
            in_dim = 15724
        elif subj[0] == 'subj02':
            in_dim = 14278
        elif subj[0] == 'subj05':
            in_dim = 13039
        elif subj[0] == 'subj07':
            in_dim = 12682

        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h),
            *[item() for item in act_and_norm],
            nn.Dropout(0.5),
        )
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                *[item() for item in act_and_norm],
                nn.Dropout(0.15)
            ) for _ in range(n_blocks)
        ])
        self.lin1 = nn.Linear(h, out_dim, bias=True)
        self.n_blocks = n_blocks
        self.clip_size = clip_size
        
        self.use_projector = use_projector
        if use_projector:
            self.projector = nn.Sequential(
                nn.LayerNorm(clip_size),
                nn.GELU(),
                nn.Linear(clip_size, 2048),
                nn.LayerNorm(2048),
                nn.GELU(),
                nn.Linear(2048, 2048),
                nn.LayerNorm(2048),
                nn.GELU(),
                nn.Linear(2048, clip_size)
            )

    def forward(self, x):
        pass

    def encode_fmri(self, x):


        x = self.lin0(x)  # bs, h
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = x.reshape(len(x), -1)
        x = self.lin1(x)
        if self.use_projector:
            return x, self.projector(x.reshape(len(x), -1, self.clip_size))
        return x


class BrainNetwork_mindeye_3d(nn.Module):
    def __init__(self, out_dim=257*1024, in_dim=76800, clip_size=1024, h=4096, n_blocks=4, norm_type='ln', act_first=False, use_projector=True, subj='all'):
    # def __init__(self, out_dim=768, in_dim=18000, clip_size=768, h=4096, n_blocks=4, norm_type='ln', act_first=False, use_projector=True,dropout_p='ddd'):
        super().__init__()
        norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm, normalized_shape=h)
        act_fn = partial(nn.ReLU, inplace=True) if norm_type == 'bn' else nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)
        # self.temp = nn.Parameter(torch.tensor(.006))

        if subj[0] == 'subj01':
            in_dim = 28224
        elif subj[0] == 'subj02':
            in_dim = 28224
        elif subj[0] == 'subj05':
            in_dim = 28224
        elif subj[0] == 'subj07':
            in_dim = 25088

        norm_layer = nn.BatchNorm3d
        self.conv1 = nn.Conv3d(1, 32, kernel_size=9, stride=3, padding=4, bias=False)
        self.bn1 = norm_layer(32)
        self.conv2 = nn.Conv3d(32, 48, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = norm_layer(48)
        self.conv3 = nn.Conv3d(48, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn3 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)


        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h),
            *[item() for item in act_and_norm],
            nn.Dropout(0.5),
        )
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                *[item() for item in act_and_norm],
                nn.Dropout(0.15)
            ) for _ in range(n_blocks)
        ])
        self.lin1 = nn.Linear(h, out_dim, bias=True)
        self.n_blocks = n_blocks
        self.clip_size = clip_size
        
        self.use_projector = use_projector
        if use_projector:
            self.projector = nn.Sequential(
                nn.LayerNorm(clip_size),
                nn.GELU(),
                nn.Linear(clip_size, 2048),
                nn.LayerNorm(2048),
                nn.GELU(),
                nn.Linear(2048, 2048),
                nn.LayerNorm(2048),
                nn.GELU(),
                nn.Linear(2048, clip_size)
            )
        
    def forward(self, x):
        pass

    def encode_fmri(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = x.reshape(len(x),-1)

        x = self.lin0(x)  # bs, h
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = x.reshape(len(x), -1)
        x = self.lin1(x)
        if self.use_projector:
            return x, self.projector(x.reshape(len(x), -1, self.clip_size))
        return x


class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResBlock1D, self).__init__()
        padding = kernel_size // 2  # 保证卷积后数据长度不变
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.stride = stride

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResBlock3D, self).__init__()
        padding = kernel_size // 2  # 保证卷积后数据尺寸不变
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.stride = stride

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class BrainNetwork_1dcnn_3d(nn.Module):
    def __init__(self, out_dim=257*1024, in_dim=76800, clip_size=1024, h=4096, n_blocks=8, norm_type='ln', act_first=False, use_projector=True, subj='all'):
    # def __init__(self, out_dim=768, in_dim=18000, clip_size=768, h=4096, n_blocks=4, norm_type='ln', act_first=False, use_projector=True,dropout_p='ddd'):
        super().__init__()
        norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm, normalized_shape=h)
        act_fn = partial(nn.ReLU, inplace=True) if norm_type == 'bn' else nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)
        # self.temp = nn.Parameter(torch.tensor(.006))

        if subj[0] == 'subj01':
            in_dim = 28224
        elif subj[0] == 'subj02':
            in_dim = 28224
        elif subj[0] == 'subj05':
            in_dim = 28224
        elif subj[0] == 'subj07':
            in_dim = 25088

        norm_layer = nn.BatchNorm3d
        self.conv1 = nn.Conv3d(1, 32, kernel_size=9, stride=3, padding=4, bias=False)
        self.bn1 = norm_layer(32)
        self.conv2 = nn.Conv3d(32, 48, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = norm_layer(48)
        self.conv3 = nn.Conv3d(48, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn3 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)


        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h),
            *[item() for item in act_and_norm],
            nn.Dropout(0.5),
        )
        self.conv01 = nn.Conv1d(1, 64,kernel_size=13, stride=6)
        self.bn01 = nn.BatchNorm1d(64)
        self.conv02 = nn.Conv1d(64, 128,kernel_size=11, stride=5)
        self.bn02 = nn.BatchNorm1d(128)
        self.conv03 = nn.Conv1d(128, 256,kernel_size=9, stride=4)
        self.bn03 = nn.BatchNorm1d(256)
        self.conv04 = nn.Conv1d(256, 512,kernel_size=7, stride=3)
        self.bn04 = nn.BatchNorm1d(512)
        self.cnn_1d = nn.ModuleList([
            nn.Sequential(
                ResBlock1D(512, 512),
                nn.Dropout(0.05)
            ) for _ in range(n_blocks)
        ])

        self.lin1 = nn.Linear(4608, out_dim, bias=True)
        self.n_blocks = n_blocks
        self.clip_size = clip_size
        
        self.use_projector = use_projector
        if use_projector:
            self.projector = nn.Sequential(
                nn.LayerNorm(clip_size),
                nn.GELU(),
                nn.Linear(clip_size, 2048),
                nn.LayerNorm(2048),
                nn.GELU(),
                nn.Linear(2048, 2048),
                nn.LayerNorm(2048),
                nn.GELU(),
                nn.Linear(2048, clip_size)
            )
        
    def forward(self, x):
        pass

    def encode_fmri(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = x.reshape(len(x),-1)

        x = self.lin0(x)  # bs, h
        x = x.reshape(len(x),1,-1)

        x = self.conv01(x)
        x = self.bn01(x)
        x = self.conv02(x)
        x = self.bn02(x)
        x = self.conv03(x)
        x = self.bn03(x)
        x = self.conv04(x)
        x = self.bn04(x)

        for res_block in range(self.n_blocks):
            x = self.cnn_1d[res_block](x)

        x = x.reshape(len(x), -1)

        x = self.lin1(x)
        if self.use_projector:
            return x, self.projector(x.reshape(len(x), -1, self.clip_size))
        return x


class BrainNetwork_3dcnn_3d(nn.Module):
    def __init__(self, out_dim=257*1024, in_dim=76800, clip_size=1024, h=4096, n_blocks=8, norm_type='ln', act_first=False, use_projector=True, subj='all'):
    # def __init__(self, out_dim=768, in_dim=18000, clip_size=768, h=4096, n_blocks=4, norm_type='ln', act_first=False, use_projector=True,dropout_p='ddd'):
        super().__init__()
        norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm, normalized_shape=h)
        act_fn = partial(nn.ReLU, inplace=True) if norm_type == 'bn' else nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)
        # self.temp = nn.Parameter(torch.tensor(.006))

        if subj[0] == 'subj01':
            in_dim = 28224
        elif subj[0] == 'subj02':
            in_dim = 28224
        elif subj[0] == 'subj05':
            in_dim = 28224
        elif subj[0] == 'subj07':
            in_dim = 25088

        norm_layer = nn.BatchNorm3d
        self.conv1 = nn.Conv3d(1, 32, kernel_size=9, stride=3, padding=4, bias=False)
        self.bn1 = norm_layer(32)
        self.conv2 = nn.Conv3d(32, 48, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = norm_layer(48)
        self.conv3 = nn.Conv3d(48, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn3 = norm_layer(64)

        self.conv4 = nn.Conv3d(64, 90, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn4 = norm_layer(90)
        self.conv5 = nn.Conv3d(90, 150, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn5 = norm_layer(150)
        self.relu = nn.ReLU(inplace=True)


        self.cnn_3d = nn.ModuleList([
            nn.Sequential(
                ResBlock3D(150, 150),
                nn.Dropout(0.15)
            ) for _ in range(n_blocks)
        ])

        self.lin1 = nn.Linear(4050, out_dim, bias=True)
        self.n_blocks = n_blocks
        self.clip_size = clip_size
        
        self.use_projector = use_projector
        if use_projector:
            self.projector = nn.Sequential(
                nn.LayerNorm(clip_size),
                nn.GELU(),
                nn.Linear(clip_size, 2048),
                nn.LayerNorm(2048),
                nn.GELU(),
                nn.Linear(2048, 2048),
                nn.LayerNorm(2048),
                nn.GELU(),
                nn.Linear(2048, clip_size)
            )
        
    def forward(self, x):
        pass

    def encode_fmri(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)


        for res_block in range(self.n_blocks):
            x = self.cnn_3d[res_block](x)

        x = x.reshape(len(x), -1)
        x = self.lin1(x)
        if self.use_projector:
            return x, self.projector(x.reshape(len(x), -1, self.clip_size))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class BrainNetwork_transformer_3d(nn.Module):
    def __init__(self, out_dim=257*1024, in_dim=76800, clip_size=1024, h=4096, n_blocks=4, norm_type='ln', act_first=False, use_projector=True, subj='all'):
    # def __init__(self, out_dim=768, in_dim=18000, clip_size=768, h=4096, n_blocks=4, norm_type='ln', act_first=False, use_projector=True,dropout_p='ddd'):
        super().__init__()
        norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm, normalized_shape=h)
        act_fn = partial(nn.ReLU, inplace=True) if norm_type == 'bn' else nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)
        # self.temp = nn.Parameter(torch.tensor(.006))

        if subj[0] == 'subj01':
            in_dim = 28224
        elif subj[0] == 'subj02':
            in_dim = 28224
        elif subj[0] == 'subj05':
            in_dim = 28224
        elif subj[0] == 'subj07':
            in_dim = 25088

        norm_layer = nn.BatchNorm3d
        self.conv1 = nn.Conv3d(1, 32, kernel_size=9, stride=3, padding=4, bias=False)
        self.bn1 = norm_layer(32)
        self.conv2 = nn.Conv3d(32, 48, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = norm_layer(48)
        self.conv3 = nn.Conv3d(48, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn3 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)


        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h),
            *[item() for item in act_and_norm],
            nn.Dropout(0.5),
        )

        self.transformer=Transformer(width=256, layers=24, heads=8)

        self.lin1 = nn.Linear(h, out_dim, bias=True)
        self.n_blocks = n_blocks
        self.clip_size = clip_size
        
        self.use_projector = use_projector
        if use_projector:
            self.projector = nn.Sequential(
                nn.LayerNorm(clip_size),
                nn.GELU(),
                nn.Linear(clip_size, 2048),
                nn.LayerNorm(2048),
                nn.GELU(),
                nn.Linear(2048, 2048),
                nn.LayerNorm(2048),
                nn.GELU(),
                nn.Linear(2048, clip_size)
            )
        
    def forward(self, x):
        pass

    def encode_fmri(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = x.reshape(len(x),-1)

        x = self.lin0(x)  # bs, h 4096

        x=x.reshape(len(x),16,256) # b l c

        x=x.permute(1, 0, 2)# l b c
        x=self.transformer(x)
        x=x.permute(1, 0, 2)# b l c

        x = x.reshape(len(x),-1)
        x = self.lin1(x)

        if self.use_projector:
            return x, self.projector(x.reshape(len(x), -1, self.clip_size))
        return x


class BrainNetwork_mindeye_3d_1d(nn.Module):
    def __init__(self, out_dim=257*1024, in_dim=86827, clip_size=1024, h=4096, n_blocks=4, norm_type='ln', act_first=False, use_projector=True, subj='all'):
    # def __init__(self, out_dim=768, in_dim=18000, clip_size=768, h=4096, n_blocks=4, norm_type='ln', act_first=False, use_projector=True,dropout_p='ddd'):
        super().__init__()
        norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm, normalized_shape=h)
        act_fn = partial(nn.ReLU, inplace=True) if norm_type == 'bn' else nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)
        # self.temp = nn.Parameter(torch.tensor(.006))

        if subj[0] == 'subj01':
            in_dim = 28224
        elif subj[0] == 'subj02':
            in_dim = 28224
        elif subj[0] == 'subj05':
            in_dim = 28224
        elif subj[0] == 'subj07':
            in_dim = 25088
        
        self.global_avg_pool_pre = nn.AvgPool1d(kernel_size=50, stride=20)


        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h),
            *[item() for item in act_and_norm],
            nn.Dropout(0.5),
        )
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                *[item() for item in act_and_norm],
                nn.Dropout(0.15)
            ) for _ in range(n_blocks)
        ])
        self.lin1 = nn.Linear(h, out_dim, bias=True)
        self.n_blocks = n_blocks
        self.clip_size = clip_size
        
        self.use_projector = use_projector
        if use_projector:
            self.projector = nn.Sequential(
                nn.LayerNorm(clip_size),
                nn.GELU(),
                nn.Linear(clip_size, 2048),
                nn.LayerNorm(2048),
                nn.GELU(),
                nn.Linear(2048, 2048),
                nn.LayerNorm(2048),
                nn.GELU(),
                nn.Linear(2048, clip_size)
            )
        
    def forward(self, x):
        pass

    def encode_fmri(self, x):

        x = x.reshape(len(x),-1)

        x = self.global_avg_pool_pre(x)

        x = self.lin0(x)  # bs, h
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = x.reshape(len(x), -1)
        x = self.lin1(x)
        if self.use_projector:
            return x, self.projector(x.reshape(len(x), -1, self.clip_size))
        return x


