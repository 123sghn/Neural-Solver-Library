import torch
import torch.nn as nn
from kappautils.param_checking import to_3tuple

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from layers.UPT_Blocks import RansGridConvnext, RansPerceiver_Encoder, TransformerModel, RansPerceiver_Decoder


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'UPT'
        self.args = args

        self.input_shape = (None, self.args.shapelist[1])
        self.output_shape = (None, self.args.out_dim)

        # grid_encoder
        self.grid_encoder = RansGridConvnext(patch_size=2, 
                                             kernel_size=3,
                                             depthwise=False, 
                                             global_response_norm=True,
                                             depths=[2, 2, 2],
                                             dims=[192, 384, 768],
                                             upsample_size=None,
                                             upsample_mode='nearest',
                                             concat_pos_to_sdf=self.args.concat_pos_to_sdf,
                                             resolution=to_3tuple(self.args.grid_resolution)
                                             )
        # mesh_encoder
        self.mesh_encoder = RansPerceiver_Encoder(num_output_tokens=args.num_output_tokens,
                                                 add_type_token=True,
                                                 init_weights='truncnormal',
                                                 dim=args.n_hidden,
                                                 num_attn_heads=args.n_heads,
                                                 input_shape=self.input_shape,
                                                 fun_dim=self.args.fun_dim
                                                 )
        # latent
        self.latent = TransformerModel(init_weights='truncnormal',
                                        drop_path_rate=0.2,
                                        drop_path_decay=False,
                                        dim=args.n_hidden,
                                        num_attn_heads=args.n_heads,
                                        depth=args.n_layers,
                                        input_shape=self.mesh_encoder.output_shape,
                                        condition_dim=None
                                        )
        # decoder
        self.decoder = RansPerceiver_Decoder(init_weights='truncnormal',
                                            dim=args.n_hidden,
                                            num_attn_heads=args.n_heads,
                                            input_shape=self.latent.output_shape,
                                            ndim=self.input_shape[1],
                                            output_shape=self.output_shape,
                                            fun_dim=self.args.fun_dim
                                            )

    def forward(self, mesh_pos, query_pos, batch_idx, unbatch_idx, unbatch_select, sdf=None):
        if sdf != None:
            # encode data
            grid_embed = self.grid_encoder(sdf)
            mesh_embed = self.mesh_encoder(mesh_pos=mesh_pos, batch_idx=batch_idx)
            embed = torch.concat([grid_embed, mesh_embed], dim=1)
        else:
            # encode data
            embed = self.mesh_encoder(mesh_pos=mesh_pos, batch_idx=batch_idx)

        # propagate
        propagated = self.latent(embed)

        # decode
        x_hat = self.decoder(propagated, query_pos=query_pos, unbatch_idx=unbatch_idx, unbatch_select=unbatch_select)

        return x_hat