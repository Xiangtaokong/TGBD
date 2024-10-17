from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):


        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x_fea=x

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x,x_fea


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()


        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):

        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text, tt):
        if tt=='original':
            image_features = self.encode_image(image)
            text_features = self.encode_text(text)

            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # print(image_features)

            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            # shape = [global_batch_size, global_batch_size]
        else:
            image_features = image
            text_features = text

            image_features=image_features.view(image_features.size(0),-1)
            text_features=text_features.view(text_features.size(0),-1)

            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # print(image_features)

            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()
        if isinstance(l, nn.AdaptiveAvgPool1d):
            l = l.half()


        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()

# class BrainNetwork(nn.Module):
#     def __init__(self, h=4096, in_dim=20000, out_dim=257*1024, seq_len=1, n_blocks=4, drop=.15, clip_size=1024):
#         super().__init__()
#         self.seq_len = seq_len
#         self.h = h
#         self.clip_size = clip_size
#         self.mixer_blocks1 = nn.ModuleList([
#             self.mixer_block1(h, drop) for _ in range(n_blocks)
#         ])
#         self.mixer_blocks2 = nn.ModuleList([
#             self.mixer_block2(seq_len, drop) for _ in range(n_blocks)
#         ])
        
#         # Output linear layer
#         self.backbone_linear = nn.Linear(h * seq_len, out_dim, bias=True) 
#         # self.clip_proj = self.projector(clip_size, clip_size, h=clip_size)
        
#         self.input_proj = self.projector(in_dim, h, h=4096)
            
#     def projector(self, in_dim, out_dim, h=4096):
#         return nn.Sequential(
#             nn.LayerNorm(in_dim),
#             nn.GELU(),
#             nn.Linear(in_dim, h),
#             nn.LayerNorm(h),
#             nn.GELU(),
#             nn.Linear(h, h),
#             nn.LayerNorm(h),
#             nn.GELU(),
#             nn.Linear(h, out_dim)
#         )
    
#     def mlp(self, in_dim, out_dim, drop):
#         return nn.Sequential(
#             nn.Linear(in_dim, out_dim),
#             nn.GELU(),
#             nn.Dropout(drop),
#             nn.Linear(out_dim, out_dim),
#         )
    
#     def mixer_block1(self, h, drop):
#         return nn.Sequential(
#             nn.LayerNorm(h),
#             self.mlp(h, h, drop),  # Token mixing
#         )

#     def mixer_block2(self, seq_len, drop):
#         return nn.Sequential(
#             nn.LayerNorm(seq_len),
#             self.mlp(seq_len, seq_len, drop)  # Channel mixing
#         )
        
#     def forward(self, x):
#         # make empty tensors
#         c,b = torch.Tensor([0.]), torch.Tensor([[0.],[0.]])
        
#         # Mixer blocks
#         residual1 = x
#         residual2 = x.permute(0,2,1)
#         for block1, block2 in zip(self.mixer_blocks1,self.mixer_blocks2):
#             x = block1(x) + residual1
#             residual1 = x
#             x = x.permute(0,2,1)
            
#             x = block2(x) + residual2
#             residual2 = x
#             x = x.permute(0,2,1)
            
#         x = x.reshape(x.size(0), -1)
#         backbone = self.backbone_linear(x).reshape(len(x), -1, self.clip_size)
#         # if self.clip_scale>0:
#         #     c = self.clip_proj(backbone)

#         # if self.blurry_recon:
#         #     b = self.blin1(x)
#         #     b = self.bdropout(b)
#         #     b = b.reshape(b.shape[0], -1, 7, 7).contiguous()
#         #     b = self.bnorm(b)
#         #     b_aux = self.b_maps_projector(b).flatten(2).permute(0,2,1)
#         #     b_aux = b_aux.view(len(b_aux), 49, 512)
#         #     b = (self.bupsampler(b), b_aux)
        
#         return backbone

#     def encode_fmri(self, fmri):
#         x=self.input_proj(fmri) 
#         print(x.shape)
#         residual1 = x
#         residual2 = x.permute(0,2,1)
#         for block1, block2 in zip(self.mixer_blocks1,self.mixer_blocks2):
#             x = block1(x) + residual1
#             residual1 = x
#             x = x.permute(0,2,1)
            
#             x = block2(x) + residual2
#             residual2 = x
#             x = x.permute(0,2,1)
            
#         x = x.reshape(x.size(0), -1)
#         print(x.shape)
#         backbone = self.backbone_linear(x).reshape(len(x), -1, self.clip_size)
#         print(backbone.shape)

#         x_fea=backbone
       
#         return x_fea,x_fea
    

class Mind_c_mlp(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 dropout_p: float,
                 ):
        super().__init__()

        # N*15000-18000
        # 目标 77*768

        self.mlp1 = self.mlp(in_dim=18000,out_dim=8192)
        self.mlp11 = self.mlp(in_dim=8192,out_dim=8192)
        self.mlp12 = self.mlp(in_dim=8192,out_dim=8192)
        self.mlp2 = self.mlp(in_dim=8192,out_dim=4096)
        self.mlp21 = self.mlp(in_dim=4096,out_dim=4096)
        self.mlp22 = self.mlp(in_dim=4096,out_dim=4096)
        self.mlp3 = self.mlp(in_dim=4096,out_dim=2048)
        self.mlp4 = self.mlp(in_dim=2048,out_dim=1024)
        self.mlp5 = self.mlp(in_dim=1024,out_dim=768)

        # self.conv3 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=3, padding=1)
        # self.global_avg_pool3 = nn.AdaptiveAvgPool1d(1024)

        # self.conv4 = nn.Conv1d(in_channels=77, out_channels=1, kernel_size=3, padding=1)

        input_resolution=image_resolution
        patch_size=vision_patch_size
        width=vision_width
        vision_heads = vision_width // 64
        layers=vision_layers
        heads=vision_heads
        output_dim=embed_dim

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))

        self.transformer = Transformer(width, layers, heads)
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.initialize_parameters()

        self.dropout = nn.Dropout(p=dropout_p)

        self.dtype=torch.float16

    def initialize_parameters(self):

        nn.init.normal_(self.positional_embedding, std=0.01)


        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def mlp(self, in_dim, out_dim, drop=0.15):
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
        )


    def encode_fmri(self, fmri):

        x = fmri.type(self.dtype)
        x = self.mlp1(x)  # [batch_size, n_ctx, d_model]
        x = self.mlp11(x)
        x = self.mlp12(x)
        x = self.mlp2(x)  # [batch_size, n_ctx, d_model]
        # x = self.conv3(x)  # [batch_size, n_ctx, d_model]
        # x = self.global_avg_pool3(x)  # [batch_size, n_ctx, d_model]
        # x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # x = x + self.positional_embedding.type(self.dtype)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.mlp21(x)
        x = self.mlp22(x)                
        x = self.mlp3(x) 
        x = self.mlp4(x) 
        x = self.mlp5(x) 

        x_fea=x
       
        return x_fea,x_fea
    
    def encode_fmri_dropout(self, fmri):

        x = fmri.type(self.dtype)
        x = self.conv1(x)  # [batch_size, n_ctx, d_model]
        x = self.dropout(x)
        x = self.global_avg_pool1(x)  # [batch_size, n_ctx, d_model]
        x = self.conv2(x)  # [batch_size, n_ctx, d_model]
        x = self.dropout(x)
        x = self.global_avg_pool2(x)  # [batch_size, n_ctx, d_model]
        x = self.conv3(x)  # [batch_size, n_ctx, d_model]
        x = self.dropout(x)
        x = self.global_avg_pool3(x)  # [batch_size, n_ctx, d_model]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x_fea=x
       
        return x_fea,x_fea


    def forward(self, image, text):

        print(ddd)

        return logits_per_image, logits_per_text


class Mind_c(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 dropout_p: float,
                 ):
        super().__init__()

        # N*15000-18000
        # 目标 77*768


        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.global_avg_pool1 = nn.AdaptiveAvgPool1d(4096)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.global_avg_pool2 = nn.AdaptiveAvgPool1d(2048)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.global_avg_pool3 = nn.AdaptiveAvgPool1d(1024)

        self.conv4 = nn.Conv1d(in_channels=77, out_channels=1, kernel_size=3, padding=1)

        input_resolution=image_resolution
        patch_size=vision_patch_size
        width=vision_width
        vision_heads = vision_width // 64
        layers=vision_layers
        heads=vision_heads
        output_dim=embed_dim

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))

        self.transformer = Transformer(width, layers, heads)
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.initialize_parameters()

        self.dropout = nn.Dropout(p=dropout_p)

        self.dtype=torch.float16

    def initialize_parameters(self):

        nn.init.normal_(self.positional_embedding, std=0.01)


        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask


    def encode_fmri(self, fmri):


        x = fmri.type(self.dtype)
        x = self.conv1(x)  # [batch_size, n_ctx, d_model]
        x = self.global_avg_pool1(x)  # [batch_size, n_ctx, d_model]
        x = self.conv2(x)  # [batch_size, n_ctx, d_model]
        x = self.global_avg_pool2(x)  # [batch_size, n_ctx, d_model]
        x = self.conv3(x)  # [batch_size, n_ctx, d_model]
        x = self.global_avg_pool3(x)  # [batch_size, n_ctx, d_model]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x_fea=x
       
        return x_fea,x_fea
    
    def encode_fmri_dropout(self, fmri):

        
        x = fmri.type(self.dtype)
        x = self.dropout(x)
        x = self.conv1(x)  # [batch_size, n_ctx, d_model]
        #x = self.dropout(x)
        x = self.global_avg_pool1(x)  # [batch_size, n_ctx, d_model]
        x = self.conv2(x)  # [batch_size, n_ctx, d_model]
        #x = self.dropout(x)
        x = self.global_avg_pool2(x)  # [batch_size, n_ctx, d_model]
        x = self.conv3(x)  # [batch_size, n_ctx, d_model]
        #x = self.dropout(x)
        x = self.global_avg_pool3(x)  # [batch_size, n_ctx, d_model]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x_fea=x
       
        return x_fea,x_fea


    def forward(self, image, text):

        print(ddd)

        return logits_per_image, logits_per_text

def build_mindc_model(state_dict: dict,dropout_p='ff'):


    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    embed_dim = state_dict["text_projection"].shape[1]

    model = Mind_c(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,dropout_p,
    )
    # model = Mind_c_mlp(
    #     embed_dim,
    #     image_resolution, vision_layers, vision_width, vision_patch_size,dropout_p,
    # )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        
        if key in state_dict:
            del state_dict[key]

    for k,v in list(state_dict.items()):
        if "visual" in k:
            del state_dict[k]
    
    convert_weights(model)
    # model.load_state_dict(state_dict,strict=False)
    convert_weights(model)
    model=model.half()

    # model=BrainNetwork()
    # model=model.half()
    
    return model


# class Mind_c(nn.Module):
#     def __init__(self,
#                  embed_dim: int,
#                  # text
#                  context_length: int,
#                  vocab_size: int,
#                  transformer_width: int,
#                  transformer_heads: int,
#                  transformer_layers: int
#                  ):
#         super().__init__()

#         # N*15000-18000
#         # 目标 77*768


#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=24, kernel_size=3, padding=1)
#         self.global_avg_pool1 = nn.AdaptiveAvgPool1d(4096)
#         self.conv2 = nn.Conv1d(in_channels=24, out_channels=48, kernel_size=3, padding=1)
#         self.global_avg_pool2 = nn.AdaptiveAvgPool1d(2048)
#         self.conv3 = nn.Conv1d(in_channels=48, out_channels=77, kernel_size=3, padding=1)
#         self.global_avg_pool3 = nn.AdaptiveAvgPool1d(768)
#         self.conv4 = nn.Conv1d(in_channels=77, out_channels=1, kernel_size=3, padding=1)
 

#         self.context_length = context_length

#         self.transformer = Transformer(
#             width=transformer_width,
#             layers=transformer_layers,
#             heads=transformer_heads,
#             attn_mask=self.build_attention_mask()
#         )

#         self.vocab_size = vocab_size
#         self.token_embedding = nn.Embedding(vocab_size, transformer_width)
#         self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
#         self.ln_final = LayerNorm(transformer_width)

#         # self.ln_post = LayerNorm(width)
#         # self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

#         self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

#         self.initialize_parameters()

#         self.dtype=torch.float16

#     def initialize_parameters(self):
#         nn.init.normal_(self.token_embedding.weight, std=0.02)
#         nn.init.normal_(self.positional_embedding, std=0.01)


#         proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
#         attn_std = self.transformer.width ** -0.5
#         fc_std = (2 * self.transformer.width) ** -0.5
#         for block in self.transformer.resblocks:
#             nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
#             nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
#             nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
#             nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

#         if self.text_projection is not None:
#             nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

#     def build_attention_mask(self):
#         # lazily create causal attention mask, with full attention between the vision tokens
#         # pytorch uses additive attention mask; fill with -inf
#         mask = torch.empty(self.context_length, self.context_length)
#         mask.fill_(float("-inf"))
#         mask.triu_(1)  # zero out the lower diagonal
#         return mask


#     def encode_fmri(self, fmri):

#         x = fmri.type(self.dtype)
#         x = self.conv1(x)  # [batch_size, n_ctx, d_model]
#         x = self.global_avg_pool1(x)  # [batch_size, n_ctx, d_model]
#         x = self.conv2(x)  # [batch_size, n_ctx, d_model]
#         x = self.global_avg_pool2(x)  # [batch_size, n_ctx, d_model]
#         x = self.conv3(x)  # [batch_size, n_ctx, d_model]
#         x = self.global_avg_pool3(x)  # [batch_size, n_ctx, d_model]
#         x = x + self.positional_embedding.type(self.dtype)
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x).type(self.dtype)
#         # x.shape = [batch_size, n_ctx, transformer.width]
#         # take features from the eot embedding (eot_token is the highest number in each sequence)
#         # x = x[torch.arange(x.shape[0]), text_.argmax(dim=-1)] @ self.text_projection

#         x = self.conv4(x) @ self.text_projection
#         x = x.squeeze(1)
#         return x


#     def forward(self, image, text):
#         image_features = self.encode_image(image)
#         text_features = self.encode_text(text)

#         # normalized features
#         image_features = image_features / image_features.norm(dim=1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=1, keepdim=True)

#         # cosine similarity as logits
#         logit_scale = self.logit_scale.exp()
#         logits_per_image = logit_scale * image_features @ text_features.t()
#         logits_per_text = logits_per_image.t()

#         # shape = [global_batch_size, global_batch_size]
#         return logits_per_image, logits_per_text

# def build_mindc_model(state_dict: dict):


#     embed_dim = state_dict["text_projection"].shape[1]
#     context_length = state_dict["positional_embedding"].shape[0]
#     vocab_size = state_dict["token_embedding.weight"].shape[0]
#     transformer_width = state_dict["ln_final.weight"].shape[0]
#     transformer_heads = transformer_width // 64
#     transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

#     model = Mind_c(
#         embed_dim,
#         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
#     )

#     for key in ["input_resolution", "context_length", "vocab_size"]:
        
#         if key in state_dict:
#             del state_dict[key]

#     for k,v in list(state_dict.items()):
#         if "visual" in k:
#             del state_dict[k]
    
#     convert_weights(model)
#     model.load_state_dict(state_dict,strict=False)
#     convert_weights(model)
#     model=model.half()
    
#     return model
