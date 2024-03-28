import torch
from torch import nn
from .utils import get_1d_sincos_pos_embed_from_grid
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from .convnextv2 import Block
import math
from .utils import LayerNorm
import warnings


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate=0.):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.dropout(F.relu(layer(x)), p=self.dropout_rate) if i < self.num_layers - 1 else layer(x)
        return x

class FeatEmbedding(nn.Module):

    def __init__(self, dim_in: int, kernel_size: int, dim_out: int) -> None:
        super().__init__()
        self.dwconv = nn.AdaptiveAvgPool2d((1, 1))
        # self.dwconv = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, groups=dim_in)
        self.norm = nn.LayerNorm(dim_in)
        self.pwconv = nn.Linear(dim_in, dim_out)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.dwconv(x).squeeze(-2, -1)  # (N, Cin, H, W) -> (N, Cin)
        x = self.norm(x)
        x = self.pwconv(x)  # (N, Cin) -> (N, Cout)
        x = self.act(x)
        return x


class FeatMixer(nn.Module):

    def __init__(self, dim_f1: int, dim_f2: int = None) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        dim = dim_f1 + dim_f2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, groups=dim)
        # self.pwconv_f1 = nn.Linear(dim_f1, dim_f2)  # pointwise/1x1 convs, implemented with linear layers
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, dim*2)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.ReLU()
        self.pwconv2 = nn.Linear(dim*2, dim // 2)

    def forward(self, feat1, feat2):
        feat1 = self.pool(feat1)  # (N, D_1, H, W) -> (N, D_1, 7, 7)
        feat2 = self.pool(feat2)  # (N, D_2, H, W) -> (N, D_2, 7, 7)
        x = torch.cat([feat1, feat2], dim=1)  # (N, D_1 + D_2,7,7)
        x = self.dwconv(x).squeeze(-2, -1)  # (N, D_2, 7, 7) -> (N, D_2)
        x = self.norm(x)
        # x = x[:, self.index_shuffle1].contiguous()
        x = self.pwconv1(x)  # (N, D_1 + D_2) -> (N, (D_1 + D_2)*2)
        x = self.act(x)
        # x = x[:, self.index_shuffle2].contiguous()
        x = self.pwconv2(x)  # (N, (D_1 + D_2)*2) -> (N, (D_1 + D_2)//2)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.ones(N, N, device=x.device).triu(diagonal=1) * -1000
        # mask[0][1] = 0  # first pos (saver model's output) is always accessible
        # model._orig_mod.attn.attn0.attn.attn_weight[0,0,10,:]
        # mask[:, 0:2] = 2
        attn += mask
        attn = attn.softmax(dim=-1)

        # self.attn_weight = attn.detach()
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FFN(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        dropout=0.,
        attn_drop=0.,
        proj_drop=0.,
        last_stage=False
    ):
        super().__init__()
        self.last_stage = last_stage
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop
        )
        if not self.last_stage:
            self.norm2 = nn.LayerNorm(dim)
            self.ffn = FFN(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        if not self.last_stage:
            x = x + self.ffn(self.norm2(x))
        return x


class DMC_train(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        saver_model: nn.Module,
        input_size: list[int] | tuple[int] = (3, 224, 224),
        dim: int = 64,
        num_classes: int = 1000,
        num_attn_layer: int = 2,
        allowed_exits: list[int] = None,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.dim = dim
        if model.profiling > 0 and allowed_exits is not None:
            allowed_exits_ = allowed_exits
            allowed_exits = []
            for i in range(len(allowed_exits_)):
                if allowed_exits_[i] <= model.profiling - 1:
                    allowed_exits.append(allowed_exits_[i])
        self.model = model
        self.allowed_exits = allowed_exits
        self.saver_model = saver_model
        device = next(model.parameters()).device
        self.feat_emb = self._build_feat_emb(model, saver_model, input_size, dim, device)
        self.attn = nn.Sequential()
        for i in range(num_attn_layer):
            self.attn.add_module(
                f'attn{i}',
                TransformerBlock(
                    dim,
                    num_heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    last_stage=(i == num_attn_layer - 1)
                )
            )

        self.eeheads = nn.ModuleList(
            [nn.Linear(dim, num_classes, bias=False) for _ in range(1, self.get_max_ee_length())]
        )

        self._freeze_backbone()
        self.apply(self._init_weights)
        # self.eeheads.apply(torch.nn.init.zeros_)
        self.to(device)

    def _freeze_backbone(self):
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.saver_model.parameters():
            p.requires_grad = False

    def _build_feat_emb(
        self, model: nn.Module, saver_model: nn.Module, input_size: list[int] | tuple[int], dim: int, device: int | str
    ):
        model.eval()
        saver_model.eval()
        inp = torch.rand(input_size).unsqueeze(0).to(device)
        _, interm_feats = model(inp)
        if model.profiling == 0:
            interm_feats = interm_feats[:-1]
        if self.allowed_exits is None:
            self.allowed_exits = list(range(len(interm_feats)))
        _, interm_feats_saver_model = saver_model(inp)
        self.max_ee_length = len(interm_feats)
        feat_emb = torch.nn.ModuleList()
        for i, f in enumerate(interm_feats):
            # feat_emb.append(nn.AdaptiveAvgPool2d((1, 1)))
            if f.shape[1]==f.shape[2]:
                raise RuntimeError(f'the shape of interm feat is {f.shape}, please ensure it is in the format of (B, C, H, W)')
            if i == 0:
                feat_emb.append(
                    FeatEmbedding(interm_feats_saver_model[-1].shape[1], interm_feats_saver_model[-1].shape[2:],
                                  dim).to(device)
                )
            else:
                feat_emb.append(FeatEmbedding(f.shape[1], f.shape[2:], dim).to(device))

        model.eval()
        saver_model.eval()
        return feat_emb

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if m.weight.requires_grad:
                trunc_normal_(m.weight, std=.02)
            if m.bias is not None and m.bias.requires_grad:
                nn.init.constant_(m.bias, 0)

    def get_max_ee_length(self):
        return self.max_ee_length

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        self.saver_model.eval()

    def forward_backbone(self, x: torch.Tensor):
        with torch.no_grad():
            out, interm_feats = self.model(x)
            saver_model_out, saver_model_interm_feats = self.saver_model(x)
        interm_feats[0] = saver_model_interm_feats[-1]
        return out, saver_model_out, interm_feats

    def forward_earlyexit(self, out, saver_model_out, interm_feats):
        ee_seq = []
        for i in self.allowed_exits:
            ee_seq.append(self.feat_emb[i](interm_feats[i]))
        ee_seq = torch.stack(ee_seq, dim=1)  # (B, N, C)
        ee_seq = self.attn(ee_seq)  # (B, N, C)
        ee_out = [saver_model_out]
        for i, head_i in enumerate(self.allowed_exits):
            if head_i == 0:
                continue
            ee_out.append(self.eeheads[head_i - 1](ee_seq[:, i]) + saver_model_out)
        ee_out = torch.stack(ee_out, dim=1)
        return out, ee_out  #, mask

    def forward(self, x: torch.Tensor):
        out, saver_model_out, interm_feats = self.forward_backbone(x)
        return self.forward_earlyexit(out, saver_model_out, interm_feats)

    @torch.no_grad()
    def forward_profiling_backbone(self, x: torch.Tensor):
        out, interm_feats = self.model(x)

        return out, interm_feats  #, mask

    @torch.no_grad()
    def forward_profiling_shared(self, interm_feats: list[torch.Tensor]):
        ee_seq = []

        with torch.no_grad():
            saver_model_out, saver_model_interm_feats = self.saver_model(interm_feats[0])
        if self.model.profiling == 1:
            return None, saver_model_out
        interm_feats[0] = saver_model_interm_feats[-1]

        for i in self.allowed_exits:
            ee_seq.append(self.feat_emb[i](interm_feats[i]))
        # print(f'len(ee_seq): {len(ee_seq)}')
        ee_seq = torch.stack(ee_seq, dim=1)  # (B, N, C)
        ee_seq = self.attn(ee_seq)  # (B, N, C)
        return ee_seq, saver_model_out

    def forward_profiling_head(self, ee_seq: torch.Tensor, saver_model_out: torch.Tensor):
        if self.model.profiling == 1:
            return None
        single_exit_out = self.eeheads[-1](ee_seq[:, -1]) + saver_model_out

        return single_exit_out  #, mask
