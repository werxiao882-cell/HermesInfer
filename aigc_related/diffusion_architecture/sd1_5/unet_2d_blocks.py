from functools import partial
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import apply_freeu
from diffusers.utils.import_utils import is_torch_version
from diffusers.models.activations import get_activation
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.modeling_outputs import Transformer2DModelOutput

logger = logging.get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# 基础组件：Downsample2D / Upsample2D / ResnetBlock2D / Transformer2DModel
# ─────────────────────────────────────────────────────────────

class Downsample2D(nn.Module):
    """
    2D 下采样层（步长为 2 的卷积）。
    use_conv=True 时使用 Conv2d（stride=2），否则用 AvgPool2d。
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        self.name = name

        if use_conv:
            conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias)
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=2, stride=2)

        # 历史遗留：name='op' 时属性名为 self.conv
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels
        # padding=0 时在右/下各补一列零，避免 feature map 尺寸不整除
        if self.use_conv and self.padding == 0:
            hidden_states = F.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(hidden_states)


class Upsample2D(nn.Module):
    """
    2D 上采样层。
    先用 nearest 插值 2×，再可选地经过一个 Conv2d 平滑特征。
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding: int = 1,
        bias: bool = True,
        interpolate: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate

        conv = None
        if use_conv_transpose:
            kernel_size = kernel_size or 4
            conv = nn.ConvTranspose2d(channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias)
        elif use_conv:
            kernel_size = kernel_size or 3
            conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, hidden_states: torch.Tensor, output_size: Optional[int] = None, **kwargs) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # bfloat16 在旧版 PyTorch 上不支持 upsample_nearest2d，临时转为 float32
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16 and is_torch_version("<", "2.1"):
            hidden_states = hidden_states.to(torch.float32)

        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        if self.interpolate:
            if output_size is None:
                hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            else:
                hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        if dtype == torch.bfloat16 and is_torch_version("<", "2.1"):
            hidden_states = hidden_states.to(dtype)

        if self.use_conv:
            hidden_states = self.conv(hidden_states) if self.name == "conv" else self.Conv2d_0(hidden_states)

        return hidden_states


class ResnetBlock2D(nn.Module):
    """
    SD1.5 标准残差块：GroupNorm → SiLU → Conv → (+Time Emb) → GroupNorm → SiLU → Conv → (+Shortcut)

    Time Embedding 注入方式：
    - "default"    : hidden = hidden + time_proj(temb)           （加法注入）
    - "scale_shift": scale, shift = chunk(time_proj(temb), 2)    （仿射调制，更强）
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        time_embedding_norm: str = "default",
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm

        groups_out = groups_out or groups

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if time_embedding_norm == "default":
                self.time_emb_proj = nn.Linear(temb_channels, out_channels)
            elif time_embedding_norm == "scale_shift":
                self.time_emb_proj = nn.Linear(temb_channels, 2 * out_channels)
            else:
                raise ValueError(f"unknown time_embedding_norm: {time_embedding_norm}")
        else:
            self.time_emb_proj = None

        self.norm2 = nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = get_activation(non_linearity)

        self.use_in_shortcut = (self.in_channels != conv_2d_out_channels) if use_in_shortcut is None else use_in_shortcut
        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(in_channels, conv_2d_out_channels, kernel_size=1, stride=1, padding=0, bias=conv_shortcut_bias)

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = input_tensor

        # 第一段：Norm → 激活 → Conv
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        # 注入 Time Embedding
        if self.time_emb_proj is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift

        # 第二段：激活 → Dropout → Conv
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        # 残差连接（当通道数不一致时用 1x1 Conv 对齐）
        if self.conv_shortcut is not None:
            if self.training:
                input_tensor = input_tensor.contiguous()
            input_tensor = self.conv_shortcut(input_tensor)

        return (input_tensor + hidden_states) / self.output_scale_factor


class Transformer2DModel(nn.Module):
    """
    SD1.5 中用于 Spatial Transformer 的 2D Transformer（连续特征版本）。

    处理流程：
    1. GroupNorm + 1×1 Conv 将 [B, C, H, W] 投影到 inner_dim
    2. reshape 为序列 [B, H×W, inner_dim]
    3. N 个 BasicTransformerBlock（Self-Attention + Cross-Attention + FFN）
    4. reshape 回 [B, H, W, inner_dim] → 1×1 Conv → + residual

    BasicTransformerBlock 内部：
    - Self-Attention : Q=K=V=图像特征（学习空间位置关系）
    - Cross-Attention: Q=图像特征, K=V=文本特征（将语义注入图像）
    - FFN (GEGLU)
    """

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gradient_checkpointing = False

        # GroupNorm 归一化 + 1×1 卷积投影（conv 模式，即 use_linear_projection=False）
        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                dropout=dropout,
                cross_attention_dim=cross_attention_dim,
                activation_fn=activation_fn,
                attention_bias=attention_bias,
                only_cross_attention=only_cross_attention,
                double_self_attention=double_self_attention,
                upcast_attention=upcast_attention,
                norm_type=norm_type,
                norm_elementwise_affine=norm_elementwise_affine,
                norm_eps=norm_eps,
                attention_type=attention_type,
            )
            for _ in range(num_layers)
        ])

        self.proj_out = nn.Conv2d(inner_dim, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        # mask 转为 bias 格式：[B, seq] → [B, 1, seq]，0=保留，-10000=丢弃
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. 输入：[B, C, H, W] → Norm → 1×1Conv → reshape → [B, H×W, inner_dim]
        batch_size, _, height, width = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_in(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, inner_dim)

        # 2. N 个 Transformer Block（Self-Attn + Cross-Attn + FFN）
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, attention_mask, encoder_hidden_states,
                    encoder_attention_mask, None, cross_attention_kwargs, None,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

        # 3. 输出：[B, H×W, inner_dim] → reshape → 1×1Conv → + residual
        hidden_states = hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        hidden_states = self.proj_out(hidden_states)
        output = hidden_states + residual

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    downsample_padding: Optional[int] = None,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = 'default',
    dropout: float = 0.0,
):
    if down_block_type == 'DownBlock2D':
        return DownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == 'CrossAttnDownBlock2D':
        return CrossAttnDownBlock2D(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    raise ValueError(f'{down_block_type} does not exist.')


def get_mid_block(
    mid_block_type: str,
    temb_channels: int,
    in_channels: int,
    resnet_eps: float,
    resnet_act_fn: str,
    resnet_groups: int,
    output_scale_factor: float = 1.0,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = 'default',
    dropout: float = 0.0,
):
    if mid_block_type == 'UNetMidBlock2DCrossAttn':
        return UNetMidBlock2DCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            resnet_groups=resnet_groups,
            upcast_attention=upcast_attention,
        )
    raise ValueError(f'unknown mid_block_type: {mid_block_type}')


def get_up_block(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    prev_output_channel: int,
    temb_channels: int,
    add_upsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    resolution_idx: Optional[int] = None,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = 'default',
    dropout: float = 0.0,
) -> nn.Module:
    if up_block_type == 'UpBlock2D':
        return UpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == 'CrossAttnUpBlock2D':
        return CrossAttnUpBlock2D(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    raise ValueError(f'{up_block_type} does not exist.')


class UNetMidBlock2DCrossAttn(nn.Module):
    """
    中间块：ResNet + (Transformer + ResNet) × num_layers
    结构：resnet -> [attn -> resnet] * N
    """

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = 'default',
        resnet_act_fn: str = 'swish',
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        output_scale_factor: float = 1.0,
        cross_attention_dim: int = 1280,
        upcast_attention: bool = False,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        resnets = [ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            eps=resnet_eps,
            groups=resnet_groups,
            dropout=dropout,
            time_embedding_norm=resnet_time_scale_shift,
            non_linearity=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            pre_norm=resnet_pre_norm,
        )]
        attentions = []
        for i in range(num_layers):
            attentions.append(Transformer2DModel(
                num_attention_heads,
                out_channels // num_attention_heads,
                in_channels=out_channels,
                num_layers=transformer_layers_per_block[i],
                cross_attention_dim=cross_attention_dim,
                norm_num_groups=resnet_groups,
                upcast_attention=upcast_attention,
            ))
            resnets.append(ResnetBlock2D(
                in_channels=out_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            ))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs, attention_mask=attention_mask, encoder_attention_mask=encoder_attention_mask, return_dict=False)[0]
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
            else:
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs, attention_mask=attention_mask, encoder_attention_mask=encoder_attention_mask, return_dict=False)[0]
                hidden_states = resnet(hidden_states, temb)
        return hidden_states


class CrossAttnDownBlock2D(nn.Module):
    """
    下采样块（带交叉注意力）：(ResNet + Transformer) × num_layers + Downsample
    结构：[resnet -> attn] * N -> downsample
    Skip connections 收集每层输出，供 Decoder 端使用。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = 'default',
        resnet_act_fn: str = 'swish',
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        add_downsample: bool = True,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
    ):
        super().__init__()
        resnets = []
        attentions = []
        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            ))
            attentions.append(Transformer2DModel(
                num_attention_heads,
                out_channels // num_attention_heads,
                in_channels=out_channels,
                num_layers=transformer_layers_per_block[i],
                cross_attention_dim=cross_attention_dim,
                norm_num_groups=resnet_groups,
                only_cross_attention=only_cross_attention,
                upcast_attention=upcast_attention,
            ))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name='op')])
        else:
            self.downsamplers = None
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        additional_residuals: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        output_states = ()
        blocks = list(zip(self.resnets, self.attentions))
        for i, (resnet, attn) in enumerate(blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs, attention_mask=attention_mask, encoder_attention_mask=encoder_attention_mask, return_dict=False)[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs, attention_mask=attention_mask, encoder_attention_mask=encoder_attention_mask, return_dict=False)[0]
            # ControlNet 等扩展可在最后一层注入残差
            if i == len(blocks) - 1 and additional_residuals is not None:
                hidden_states = hidden_states + additional_residuals
            output_states = output_states + (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)
        return hidden_states, output_states


class DownBlock2D(nn.Module):
    """
    下采样块（无注意力）：ResNet × num_layers + Downsample
    用于 U-Net 最深层，此时分辨率已足够小，无需跨模态交互。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = 'default',
        resnet_act_fn: str = 'swish',
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            ))
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name='op')])
        else:
            self.downsamplers = None
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        output_states = ()
        for resnet in self.resnets:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)
            output_states = output_states + (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)
        return hidden_states, output_states


class CrossAttnUpBlock2D(nn.Module):
    """
    上采样块（带交叉注意力）：(ResNet + Transformer) × num_layers + Upsample
    每个 ResNet 的输入会与对应的 Encoder Skip Connection 在通道维度上拼接。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = 'default',
        resnet_act_fn: str = 'swish',
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
    ):
        super().__init__()
        resnets = []
        attentions = []
        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock2D(
                # Decoder 的每层输入 = 上一层输出 + Encoder Skip Connection（cat）
                in_channels=resnet_in_channels + res_skip_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            ))
            attentions.append(Transformer2DModel(
                num_attention_heads,
                out_channels // num_attention_heads,
                in_channels=out_channels,
                num_layers=transformer_layers_per_block[i],
                cross_attention_dim=cross_attention_dim,
                norm_num_groups=resnet_groups,
                only_cross_attention=only_cross_attention,
                upcast_attention=upcast_attention,
            ))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None
        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # FreeU: 可选地对 hidden_states 和 skip connection 做频域增强
        is_freeu_enabled = getattr(self, 's1', None) and getattr(self, 's2', None) and getattr(self, 'b1', None) and getattr(self, 'b2', None)
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(self.resolution_idx, hidden_states, res_hidden_states, s1=self.s1, s2=self.s2, b1=self.b1, b2=self.b2)
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs, attention_mask=attention_mask, encoder_attention_mask=encoder_attention_mask, return_dict=False)[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs, attention_mask=attention_mask, encoder_attention_mask=encoder_attention_mask, return_dict=False)[0]
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        return hidden_states


class UpBlock2D(nn.Module):
    """
    上采样块（无注意力）：ResNet × num_layers + Upsample
    对应 Encoder 最深层的 DownBlock2D，同样无需跨模态交互。
    """

    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = 'default',
        resnet_act_fn: str = 'swish',
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock2D(
                in_channels=resnet_in_channels + res_skip_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            ))
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None
        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        upsample_size: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        is_freeu_enabled = getattr(self, 's1', None) and getattr(self, 's2', None) and getattr(self, 'b1', None) and getattr(self, 'b2', None)
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(self.resolution_idx, hidden_states, res_hidden_states, s1=self.s1, s2=self.s2, b1=self.b1, b2=self.b2)
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        return hidden_states
