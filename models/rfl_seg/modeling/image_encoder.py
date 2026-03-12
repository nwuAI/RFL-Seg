


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type, List
from models.rfl_seg.modeling.common import LayerNorm2d, MLPBlock, Adapter, AugAdapter
import math


class EnhancedCNNEmbed(nn.Module):
    """Enhanced CNN embedding layer using multi-scale convolution"""

    def __init__(
            self,
            patchsize: int = 8,  # default: 8
            in_chans: int = 1,
            embed_dim: int = 768,
    ) -> None:
        super().__init__()
        downtimes = int(math.log2(patchsize))  # 3
        mid_channel = 64

        # Using Inception module as initial convolution :3 , 64
        self.inc = InceptionBlock(in_chans, mid_channel)

        self.downs = nn.ModuleList()
        for i in range(downtimes):  # 0, 1, 2
            out_channels = embed_dim if i == downtimes - 1 else mid_channel * 2  # out_channels:128->256->768

            # Using Inception module
            down = nn.Sequential(
                nn.MaxPool2d(2),
                InceptionBlock(mid_channel, out_channels)
            )

            self.downs.append(down)
            mid_channel = out_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features = []
        x = self.inc(x)  # [b,3,256,256]->[b,64,256,256]
        features.append(x)
        for down in self.downs:
            x = down(x)
            features.append(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)

        return x, features


class ChannelAttention(nn.Module):
    """Channel attention mechanism to enhance important feature channels"""

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class InceptionBlock(nn.Module):
    """Multi-scale convolution module for extracting features at different scales"""

    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        mid_channels = out_channels // 4  # 3, 64-> 3, 64

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            LayerNorm2d(mid_channels),
            nn.GELU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            LayerNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            LayerNorm2d(mid_channels),
            nn.GELU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            LayerNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=5, padding=2),
            LayerNorm2d(mid_channels),
            nn.GELU()
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            LayerNorm2d(mid_channels),
            nn.GELU()
        )

        self.ca = ChannelAttention(out_channels)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        out = self.ca(out) * out
        return out


class ImageEncoderViT(nn.Module):
    def __init__(
            self,
            img_size: int = 256,
            patch_size: int = 8,
            in_chans: int = 1,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.img_size = img_size

        # Using enhanced CNN embedding layer (multi-scale convolution)
        self.cnn_embed = EnhancedCNNEmbed(patchsize=patch_size, in_chans=3, embed_dim=embed_dim)

        # Using standard ViT embedding layer
        self.patch_embed = PatchEmbed0(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=3,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Maintain original positional encoding size (64x64) to match pretrained weights
            self.pos_embed = nn.Parameter(
                torch.zeros(1, 64, 64, embed_dim)  # Original size
            )

        self.blocks = nn.ModuleList()

        for i in range(depth):
            block = ParaBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                depth=i,
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )
        self.input_Adapter = Adapter(embed_dim)

        # Add feature fusion attention mechanism
        self.fusion_attention = ChannelAttention(embed_dim)

    # original_image: [4,1,256,256]
    def forward(self, original_image, gen_image) -> Tuple[
        torch.Tensor, List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        anomaly_image = torch.abs(original_image - gen_image)
        if original_image.size()[1] == 1:
            original_image = original_image.repeat(1, 3, 1, 1)
            anomaly_image = anomaly_image.repeat(1, 3, 1, 1)
        cnnx, cnn_feature_list = self.cnn_embed(anomaly_image)  # [B, 3, 256, 256]-> [B, 32, 32, 768]
        x = self.patch_embed(original_image)  # [B, 3, 256, 256]-> [B, 32, 32, 768]
        x = self.input_Adapter(x)  # Through feature adapter (FA)
        if self.pos_embed is not None:
            # Ensure positional encoding size matches feature map
            B, H, W, C = x.shape
            pos_embed = self.pos_embed  # [1, 64, 64, 768]

            # If sizes don't match, perform interpolation adjustment
            if H != pos_embed.size(1) or W != pos_embed.size(2):
                pos_embed = F.interpolate(
                    pos_embed.permute(0, 3, 1, 2),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
            x = x + pos_embed.repeat(B, 1, 1, 1)
        vit_feature_list = []  # Collect feature maps with window_size=0 in ViT branch
        for blk in self.blocks:
            x, cnnx = blk(x, cnnx)
            # Collect feature maps with window_size=0 in ViT branch
            if blk.window_size == 0:
                vit_feature_list.append(x.permute(0, 3, 1, 2))
        x_perm = x.permute(0, 3, 1, 2)
        cnnx_perm = cnnx.permute(0, 3, 1, 2)
        # Apply channel attention
        attention_map = self.fusion_attention(x_perm)
        cnnx_perm = attention_map * cnnx_perm
        # Fuse features
        fused_feature = x_perm + 0.5 * cnnx_perm
        x = fused_feature.permute(0, 2, 3, 1)
        x = self.neck(x.permute(0, 3, 1, 2))
        # x:[1,256,32,32], vit_features_list:[1,768,32,32]*4, anomaly_image:[1,3,256,256] ,
        # cnn_features_list:[1,64,256,256], [1,128,128,128],[1,256,64,64],[1,768,32,32]
        return x, vit_feature_list, anomaly_image, cnn_feature_list


class ParaBlock(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            input_size: Optional[Tuple[int, int]] = None,
            depth: int = 0
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

        # ------------------ new to sam----------------------
        # Space_Adapter: CBA cross-branch attention module
        # MLP_Adapter: FA (feature adapter) feature adapter (down linear projection -> GELU activation -> up linear projection)
        if self.window_size == 0:
            self.MLP_Adapter = Adapter(dim, skip_connect=False)  # new to sam, MLP-adapter, no skip connection
            self.Space_Adapter = qkvAttention(dim=dim, num_heads=num_heads)  # with skip connection
            self.refine_Adapter = SingleConv(in_channels=dim, out_channels=dim)
            self.scale = 0.5
        # ---------------------------------------------------
        self.dim = dim
        self.depth = depth
        # New: for storing intermediate feature maps
        # self.features_list = []

    # X: [1,32,32,,768], CNNX:[1,32,32,768]
    def forward(self, x: torch.Tensor, cnnx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shortcut = x
        x = self.norm1(x)  # [B, 32, 32, 768]
        # Window partition
        if self.window_size > 0:  # Other layers
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)  # x becomes [B * 4, 14, 14, 768] (4 windows), pad_hw=(32, 32)

        if self.window_size == 0:  # Layers 2,5,8,11
            sax = self.Space_Adapter(x, cnnx, cnnx)  # b h w c
            x = x + sax
            cnnx = self.refine_Adapter(cnnx.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        x = self.attn(x)

        if self.window_size > 0:  # Other layers
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x

        xn = self.norm2(x)
        x = x + self.mlp(xn)

        if self.window_size == 0:  # Layers 2,5,8,11
            x = x + self.scale * self.MLP_Adapter(xn)
        # # Store current layer's feature map (only when it's global attention)
        # if self.window_size == 0:
        #     self.features_list.append(x.detach())  # detach to avoid affecting gradient flow

        return x, cnnx


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                    input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv0 = self.qkv(x)
        qkv = qkv0.reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class qkvAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                    input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = q.shape
        q = self.q(q).reshape(B, H * W, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B * self.num_heads, H * W, -1)
        k = self.k(k).reshape(B, H * W, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B * self.num_heads, H * W, -1)
        v = self.v(v).reshape(B, H * W, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B * self.num_heads, H * W, -1)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
        windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
        attn: torch.Tensor,
        q: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
            attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=1, bias=False),
            LayerNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class SingleDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU()  # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class SingleConv(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)


# This is the position Adapter
class PostPosEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
            self,
            embed_dim: int = 768,
            ori_feature_size: int = 64,
            new_feature_size: int = 32,
    ) -> None:
        """
        Args:
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        downtimes = int(math.log2(ori_feature_size // new_feature_size))
        self.downs = nn.ModuleList()
        for i in range(downtimes):
            down = SingleDown(embed_dim, embed_dim)
            # down = nn.MaxPool2d(2)
            self.downs.append(down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B H W C -> B C H W
        x = x.permute(0, 3, 1, 2)  # [1, h, w, c]
        for down in self.downs:
            x = down(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


class PatchEmbed0(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
            self,
            kernel_size: Tuple[int, int] = (16, 16),
            stride: Tuple[int, int] = (16, 16),
            padding: Tuple[int, int] = (0, 0),
            in_chans: int = 3,
            embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=16, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, (256 + 8, 256 + 8), mode="bilinear", align_corners=False)
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x
