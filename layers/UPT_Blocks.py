import numpy as np
import einops
import torch
import torch.nn.functional as F
from torch import nn
from kappamodules.convolution import ConvNext
from kappamodules.layers import ContinuousSincosEmbed, LinearProjection
from kappamodules.transformer import PerceiverPoolingBlock, Mlp, PerceiverBlock, DitBlock, PrenormBlock
from torch_geometric.utils import to_dense_batch, unbatch
from functools import partial

################################################################
# UPT Grid Encoder
################################################################

class RansGridConvnext(nn.Module):
    def __init__(
            self,
            patch_size,
            dims,
            depths,
            kernel_size=7,
            depthwise=True,
            global_response_norm=True,
            drop_path_rate=0.,
            drop_path_decay=False,
            add_pos_tokens=False,
            upsample_size=None,
            upsample_mode="nearest",
            resolution=None,
            concat_pos_to_sdf=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.dims = dims
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.add_pos_tokens = add_pos_tokens
        self.upsample_size = upsample_size
        self.upsample_mode = upsample_mode
        self.resolution = resolution
        self.ndim = len(self.resolution)

        # sdf + grid_pos
        concat_pos_to_sdf = concat_pos_to_sdf
        if concat_pos_to_sdf:
            input_dim = 4
        else:
            input_dim = 1


        self.model = ConvNext(
            patch_size=patch_size,
            input_dim=input_dim,
            dims=dims,
            depths=depths,
            ndim=self.ndim,
            drop_path_rate=drop_path_rate,
            drop_path_decay=drop_path_decay,
            kernel_size=kernel_size,
            depthwise=depthwise,
            global_response_norm=global_response_norm,
        )


        out_resolution = [r // 2 ** (len(depths) - 1) // patch_size for r in self.resolution]
        num_output_tokens = int(np.prod(out_resolution))
        if add_pos_tokens:
            self.pos_tokens = nn.Parameter(torch.empty(size=(1, num_output_tokens, dims[-1])))
        else:
            self.pos_tokens = None
        self.type_token = nn.Parameter(torch.empty(size=(1, 1, dims[-1])))

        self.output_shape = (num_output_tokens, dims[-1])

    def forward(self, x):
        # sdf is passed as dim-last with spatial -> convert to dim-first with spatial
        x = einops.rearrange(x, "batch_size height width depth dim -> batch_size dim height width depth")
        # upsample
        if self.upsample_size is not None:
            if self.upsample_mode == "nearest":
                x = F.interpolate(x, size=self.upsample_size, mode=self.upsample_mode)
            else:
                x = F.interpolate(x, size=self.upsample_size, mode=self.upsample_mode, align_corners=True)

        # embed
        x = self.model(x)
        # flatten to tokens
        x = einops.rearrange(x, "batch_size dim height width depth -> batch_size (height width depth) dim")
        x = x + self.type_token
        if self.add_pos_tokens:
            x = x + self.pos_tokens.expand(len(x), -1, -1)
        return x
    
################################################################
# UPT Mesh Encoder
################################################################

class RansPerceiver_Encoder(nn.Module):
    def __init__(
            self,
            dim,
            num_attn_heads,
            num_output_tokens,
            add_type_token=False,
            init_weights="xavier_uniform",
            init_last_proj_zero=False,
            input_shape=None,
            fun_dim=0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_attn_heads = num_attn_heads
        self.num_output_tokens = num_output_tokens
        self.add_type_token = add_type_token
        
        self.input_shape = input_shape

        self.fun_dim = fun_dim

        # set ndim
        _, ndim = self.input_shape
        ndim = ndim - self.fun_dim

        # pos_embed
        if self.fun_dim != 0:
            self.pos_embed = ContinuousSincosEmbed(dim=dim - self.fun_dim, ndim=ndim)
        else:
            self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=ndim)

        # perceiver
        self.mlp = Mlp(in_dim=dim, hidden_dim=dim * 4, init_weights=init_weights)
        self.block = PerceiverPoolingBlock(
            dim=dim,
            num_heads=num_attn_heads,
            num_query_tokens=num_output_tokens,
            perceiver_kwargs=dict(
                init_weights=init_weights,
                init_last_proj_zero=init_last_proj_zero,
            ),
        )

        if add_type_token:
            self.type_token = nn.Parameter(torch.empty(size=(1, 1, dim,)))
        else:
            self.type_token = None

        # output shape
        self.output_shape = (num_output_tokens, dim)

    def forward(self, mesh_pos, batch_idx, mesh_edges=None):
        if self.fun_dim != 0:
            f = mesh_pos[:, -self.fun_dim:]
            mesh_pos = mesh_pos[:, :-self.fun_dim]
            x = self.pos_embed(mesh_pos)
            x = torch.cat([x, f], dim=1)
        else:
            x = self.pos_embed(mesh_pos)
        x, mask = to_dense_batch(x, batch_idx)
        if torch.all(mask):
            mask = None
        else:
            # add dimensions for num_heads and query (keys are masked)
            mask = einops.rearrange(mask, "batchsize num_nodes -> batchsize 1 1 num_nodes")

        # perceiver
        x = self.mlp(x)
        x = self.block(kv=x, attn_mask=mask)

        if self.add_type_token:
            x = x + self.type_token

        return x
    
################################################################
# UPT Latent
################################################################

class TransformerModel(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            num_attn_heads,
            drop_path_rate=0.0,
            drop_path_decay=True,
            init_weights="xavier_uniform",
            init_last_proj_zero=False,
            input_shape=None,
            condition_dim=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.depth = depth
        self.num_attn_heads = num_attn_heads
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.init_weights = init_weights
        self.init_last_proj_zero = init_last_proj_zero

        self.input_shape = input_shape
        self.condition_dim = condition_dim

        # input/output shape
        assert len(self.input_shape) == 2
        seqlen, input_dim = self.input_shape
        self.output_shape = (seqlen, dim)

        self.input_proj = LinearProjection(input_dim, dim, init_weights=init_weights)

        # blocks
        if self.condition_dim is not None:
            block_ctor = partial(DitBlock, cond_dim=self.condition_dim)
        else:
            block_ctor = PrenormBlock
        if drop_path_decay:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate] * depth
        self.blocks = nn.ModuleList([
            block_ctor(
                dim=dim,
                num_heads=num_attn_heads,
                drop_path=dpr[i],
                init_weights=init_weights,
                init_last_proj_zero=init_last_proj_zero,
            )
            for i in range(self.depth)
        ])

    def forward(self, x, condition=None, static_tokens=None):
        assert x.ndim == 3

        # concat static tokens
        if static_tokens is not None:
            x = torch.cat([static_tokens, x], dim=1)

        # input projection
        x = self.input_proj(x)

        # apply blocks
        blk_kwargs = dict(cond=condition) if condition is not None else dict()
        for blk in self.blocks:
            x = blk(x, **blk_kwargs)

        # remove static tokens
        if static_tokens is not None:
            num_static_tokens = static_tokens.size(1)
            x = x[:, num_static_tokens:]

        return x
    
################################################################
# UPT Decoder
################################################################

class RansPerceiver_Decoder(nn.Module):
    def __init__(
            self,
            dim,
            num_attn_heads,
            init_weights="xavier_uniform",
            init_last_proj_zero=False,
            use_last_norm=False,
            input_shape=None,
            ndim=None,
            output_shape=None,
            fun_dim=0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_attn_heads = num_attn_heads
        self.use_last_norm = use_last_norm

        self.input_shape = input_shape
        self.ndim = ndim - fun_dim
        self.output_shape = output_shape

        self.fun_dim = fun_dim

        # input projection
        _, input_dim = self.input_shape
        self.proj = LinearProjection(input_dim, dim, init_weights=init_weights)

        # query tokens (create them from a positional embedding)
        if self.fun_dim != 0:
            self.pos_embed = ContinuousSincosEmbed(dim=dim - self.fun_dim, ndim=self.ndim)
        else:
            self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=self.ndim)
        self.query_mlp = Mlp(in_dim=dim, hidden_dim=dim, init_weights=init_weights)

        # latent to pixels
        self.perceiver = PerceiverBlock(
            dim=dim,
            num_heads=num_attn_heads,
            init_last_proj_zero=init_last_proj_zero,
            init_weights=init_weights,
        )
        _, output_dim = self.output_shape
        self.norm = nn.LayerNorm(dim, eps=1e-6) if use_last_norm else nn.Identity()
        self.pred = LinearProjection(dim, output_dim, init_weights=init_weights)

    def forward(self, x, query_pos, unbatch_idx, unbatch_select):
        # input projection
        x = self.proj(x)

        # create query
        if self.fun_dim != 0:
            f = query_pos[:, :, -self.fun_dim:]
            query_pos = query_pos[:, :, :-self.fun_dim]
            query_pos_embed = self.pos_embed(query_pos)
            query_pos_embed = torch.cat([query_pos_embed, f], dim=2)
        else:
            query_pos_embed = self.pos_embed(query_pos)
        query = self.query_mlp(query_pos_embed)

        # decode
        x = self.perceiver(q=query, kv=x)
        x = self.norm(x)
        x = self.pred(x)

        # dense tensor (batch_size, max_num_points, dim) -> sparse tensor (batch_size * num_points, dim)
        x = einops.rearrange(x, "batch_size max_num_points dim -> (batch_size max_num_points) dim")
        unbatched = unbatch(x, batch=unbatch_idx)
        x = torch.concat([unbatched[i] for i in unbatch_select])

        return x