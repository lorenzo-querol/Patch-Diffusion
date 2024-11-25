from abc import abstractmethod

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

from .attention import SpatialTransformer


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        # self.positional_embedding = nn.Parameter(
        #     torch.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        # )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        # x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class SimpleAttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)

        ch = c
        q = k = v = x
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        x = torch.einsum("bts,bcs->bct", weight, v)

        x = x.reshape(b, -1, x.shape[-1])

        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout_rate: the rate of dropout_rate.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout_rate,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout_rate = dropout_rate
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        def _inner_forward(x):
            b, c, *spatial = x.shape
            x = x.reshape(b, c, -1)
            qkv = self.qkv(self.norm(x))
            h = self.attention(qkv)
            h = self.proj_out(h)
            return (x + h).reshape(b, c, *spatial)

        if self.use_checkpoint and x.requires_grad:
            return cp.checkpoint(_inner_forward, x)
        else:
            return _inner_forward(x)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


# @persistence.persistent_class
# class EBMUNet(torch.nn.Module):
#     def __init__(
#         self,
#         img_resolution,  # Image resolution at input/output.
#         in_channels,  # Number of color channels at input.
#         out_channels,  # Number of color channels at output.
#         label_dim=0,  # Number of class labels, 0 = unconditional.
#         model_channels=192,  # Base multiplier for the number of channels.
#         channel_mult=[1, 2, 3, 4],  # Per-resolution multipliers for the number of channels.
#         channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
#         num_blocks=3,  # Number of residual blocks per resolution.
#         attn_resolutions=[32, 16, 8],  # List of resolutions with self-attention.
#         dropout_rate=0.10,  #  Dropout probability of intermediate activations.
#     ):
#         super().__init__()
#         emb_channels = model_channels * channel_mult_emb
#         init = dict(init_mode="kaiming_uniform", init_weight=np.sqrt(1 / 3), init_bias=np.sqrt(1 / 3))
#         init_zero = dict(init_mode="kaiming_uniform", init_weight=0, init_bias=0)
#         block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout_rate, init=init, init_zero=init_zero)

#         # Mapping.
#         self.map_noise = PositionalEmbedding(num_channels=model_channels)
#         self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
#         self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
#         self.label_dim = label_dim
#         self.map_label = Linear(in_features=label_dim, out_features=emb_channels, bias=False, init_mode="kaiming_normal", init_weight=np.sqrt(label_dim)) if label_dim else None

#         # Encoder.
#         self.enc = torch.nn.ModuleDict()
#         cout = in_channels
#         for level, mult in enumerate(channel_mult):
#             res = img_resolution >> level
#             if level == 0:
#                 cin = cout
#                 cout = model_channels * mult
#                 self.enc[f"{res}x{res}_conv"] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
#             else:
#                 self.enc[f"{res}x{res}_down"] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
#             for idx in range(num_blocks):
#                 cin = cout
#                 cout = model_channels * mult
#                 self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
#         skips = [block.out_channels for block in self.enc.values()]

#         # Decoder.
#         self.dec = torch.nn.ModuleDict()
#         for level, mult in reversed(list(enumerate(channel_mult))):
#             res = img_resolution >> level
#             if level == len(channel_mult) - 1:
#                 self.dec[f"{res}x{res}_in0"] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
#                 self.dec[f"{res}x{res}_in1"] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
#             else:
#                 self.dec[f"{res}x{res}_up"] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
#             for idx in range(num_blocks + 1):
#                 cin = cout + skips.pop()
#                 cout = model_channels * mult
#                 self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)

#         self.out = torch.nn.Sequential(
#             GroupNorm(num_channels=cout),
#             torch.nn.SiLU(),
#             Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero),
#         )
#         self.fc = SimpleAttentionPool2d()

#     def forward(self, x, noise_labels, class_labels=None, augment_labels=None, cls_mode=False):
#         """
#         Forward pass with EBM functionality.
#         Arguments:
#             x:                  Input tensor
#             noise_labels:       Timestep embeddings
#             class_labels:       Optional class-conditioning
#             augment_labels:     Optional augmentation labels
#             cls_mode:           If True, runs in classification mode
#         """
#         if cls_mode:
#             emb = self.map_noise(noise_labels)
#             emb = silu(self.map_layer0(emb))
#             emb = self.map_layer1(emb)

#             y = torch.ones((x.shape[0], self.label_dim)).to(x.device)
#             emb += self.map_label(y)

#             # Encoder.
#             skips = []
#             for block in self.enc.values():
#                 x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
#                 skips.append(x)

#             # Decoder.
#             for block in self.dec.values():
#                 if x.shape[1] != block.in_channels:
#                     x = torch.cat([x, skips.pop()], dim=1)
#                 x = block(x, emb)

#             x = self.out(x)
#             return self.fc(x)
#         else:
#             with  torch.enable_grad():
#                 # Noise embedding
#                 emb = self.map_noise(noise_labels)
#                 emb = silu(self.map_layer0(emb))
#                 emb = self.map_layer1(emb)

#                 if class_labels is not None:
#                     emb += self.map_label(class_labels)

#                 # Separate image and coordinates
#                 image = x[:, :3]  # Assuming first 3 channels are RGB
#                 coords = x[:, 3:].detach()  # Coordinates channels, detached from gradient computation

#                 # Only make image require gradients
#                 input_tensor = torch.autograd.Variable(image, requires_grad=True)

#                 # Recombine for forward pass
#                 x = torch.cat([input_tensor, coords], dim=1)

#                 # Encoder.
#                 skips = []
#                 for block in self.enc.values():
#                     x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
#                     skips.append(x)

#                 # Decoder.
#                 for block in self.dec.values():
#                     if x.shape[1] != block.in_channels:
#                         x = torch.cat([x, skips.pop()], dim=1)
#                     x = block(x, emb)

#                 # EBM gradient computation.
#                 logits = self.out(x)
#                 logits_logsumexp = logits.logsumexp(1)

#                 if self.training:
#                     x_prime = torch.autograd.grad(logits_logsumexp.sum(), [input_tensor], create_graph=True, retain_graph=True)[0]
#                 else:
#                     x_prime = torch.autograd.grad(logits_logsumexp.sum(), [input_tensor])[0]

#                 return -1 * x_prime


class EBMUNet(nn.Module):
    """
    The full U-Net model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_blocks: number of residual blocks per downsample.
    :param attn_resolutions: a collection of downsample rates at which attention will take place. May be a set, list, or tuple.
    :param dropout_rate: The dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param label_dim: if specified (as an int), then this model will be class-conditional with `label_dim` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: The number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use a fixed channel width per attention head.
    :param (DEPRECATED) num_heads_upsample: works with num_heads to set a different number of heads for upsampling.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially increased efficiency.
    :param use_spatial_transformer: use a spatial transformer instead of attention.
    :param context_dim: the dimensionality of the context for the spatial transformer.
    :param transformer_depth: the depth of the spatial transformer.
    :param pool: the pooling method for the spatial transformer.
    """

    def __init__(
        self,
        img_resolution,
        in_channels,
        model_channels,
        out_channels,
        num_blocks,
        attn_resolutions,
        dropout_rate=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        label_dim=None,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,
        context_dim=512,
        transformer_depth=None,
        pool="attn",
    ):
        super().__init__()

        self.use_spatial_transformer = use_spatial_transformer
        self.context_dim = context_dim
        self.transformer_depth = transformer_depth
        if label_dim is not None:
            self.label_emb = nn.Embedding(label_dim + 1, context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout_rate = dropout_rate
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.label_dim = label_dim
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout_rate,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attn_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                        if not use_spatial_transformer
                        else SpatialTransformer(
                            ch,
                            num_heads,
                            num_head_channels,
                            depth=transformer_depth,
                            context_dim=context_dim,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout_rate,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout_rate,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            (
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=num_head_channels,
                    use_new_attention_order=use_new_attention_order,
                )
                if not use_spatial_transformer
                else SpatialTransformer(
                    ch,
                    num_heads,
                    num_head_channels,
                    depth=transformer_depth,
                    context_dim=context_dim,
                )
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout_rate,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout_rate,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attn_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                        if not use_spatial_transformer
                        else SpatialTransformer(
                            ch,
                            num_heads,
                            num_head_channels,
                            depth=transformer_depth,
                            context_dim=context_dim,
                        )
                    )
                if level and i == num_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout_rate,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

        self.pool = pool
        if pool == "adaptive":
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
        elif pool == "attn":
            self.fc = nn.Sequential(
                AttentionPool2d(img_resolution, out_channels, out_channels, label_dim),
            )
        elif pool == "sattn":
            self.fc = nn.Sequential(
                SimpleAttentionPool2d(),
            )

    def forward(self, x, timesteps, class_labels=None, cls_mode=False):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param class_labels: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        # If class_labels is one-hot encoded, convert to class indices
        if class_labels is not None and len(class_labels.shape) > 1:
            class_labels = torch.argmax(class_labels, dim=1)

        if cls_mode:
            hs = []
            emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
            # class_labels = class_labels * 0 + self.label_dim
            class_labels = torch.ones(x.shape[0]).long().to(x.device) * self.label_dim
            context = self.label_emb(class_labels)
            context = context[:, None, ...]

            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)
            for module in self.output_blocks:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, emb, context)
            h = h.type(x.dtype)
            logits = self.out(h)

            return self.fc(logits)

        else:
            with torch.enable_grad():
                hs = []
                emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
                if self.label_dim is not None:
                    if class_labels is None:
                        class_labels = torch.zeros(x.shape[0]).long().to(x.device) * self.label_dim
                    context = self.label_emb(class_labels)
                    context = context[:, None, ...]
                else:
                    context = None

                image = x[:, :3]  # Assuming first 3 channels are RGB
                coords = x[:, 3:].detach()  # Coordinates channels, detached from gradient computation
                input_tensor = torch.autograd.Variable(image, requires_grad=True)  # Only compute gradients w.r.t. image

                h = x.type(self.dtype)
                h = torch.cat([input_tensor, coords], dim=1)

                for module in self.input_blocks:
                    h = module(h, emb, context)
                    hs.append(h)
                h = self.middle_block(h, emb, context)
                for module in self.output_blocks:
                    h = torch.cat([h, hs.pop()], dim=1)
                    h = module(h, emb, context)
                h = h.type(x.dtype)
                logits = self.out(h)

                logits_logsumexp = logits.logsumexp(1)

                if self.training:
                    x_prime = torch.autograd.grad(
                        logits_logsumexp.sum(),
                        [input_tensor],
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                else:
                    x_prime = torch.autograd.grad(logits_logsumexp.sum(), [input_tensor])[0]

                x_prime = x_prime * -1

            return x_prime
