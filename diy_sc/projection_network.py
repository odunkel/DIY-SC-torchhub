import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from diy_sc.resnet import ResNet, BottleneckBlock


""""
Module adapted from https://github.com/Junyi42/GeoAware-SC/blob/master/model_utils/projection_network.py
"""

__all__ = ['Model', 'model']

class AggregationNetwork(nn.Module):
    """
    Module for aggregating feature maps across time and space.
    Design inspired by the Feature Extractor from ODISE (Xu et. al., CVPR 2023).
    https://github.com/NVlabs/ODISE/blob/5836c0adfcd8d7fd1f8016ff5604d4a31dd3b145/odise/modeling/backbone/feature_extractor.py
    """
    def __init__(
            self, 
            device, 
            feature_dims=[640,1280,1280,768],
            projection_dim=384,
            num_norm_groups=32,
            save_timestep=[1],
            kernel_size = [1,3,1],
            contrastive_temp = 10,
            feat_map_dropout = 0.0,
        ):
        super().__init__()
        self.skip_connection = True
        self.feat_map_dropout = feat_map_dropout
        self.azimuth_embedding = None
        self.pos_embedding = None
        self.bottleneck_layers = nn.ModuleList()
        self.feature_dims = feature_dims
        # For CLIP symmetric cross entropy loss during training
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.self_logit_scale = nn.Parameter(torch.ones([]) * np.log(contrastive_temp))
        self.device = device
        self.save_timestep = save_timestep
        
        self.mixing_weights_names = []
        for l, feature_dim in enumerate(self.feature_dims):
            bottleneck_layer = nn.Sequential(
                *ResNet.make_stage(
                    BottleneckBlock,
                    num_blocks=1,
                    in_channels=feature_dim,
                    bottleneck_channels=projection_dim // 4,
                    out_channels=projection_dim,
                    norm="GN",
                    num_norm_groups=num_norm_groups,
                    kernel_size=kernel_size
                )
            )
            self.bottleneck_layers.append(bottleneck_layer)
            for t in save_timestep:
                # 1-index the layer name following prior work
                self.mixing_weights_names.append(f"timestep-{save_timestep}_layer-{l+1}")
        self.last_layer = None
        self.bottleneck_layers = self.bottleneck_layers.to(device)
        mixing_weights = torch.ones(len(self.bottleneck_layers) * len(save_timestep))
        self.mixing_weights = nn.Parameter(mixing_weights.to(device))
        # count number of parameters
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(f"AggregationNetwork has {num_params} parameters.")
    
    def load_pretrained_weights(self, pretrained_dict):
        custom_dict = self.state_dict()

        # Handle size mismatch
        if 'mixing_weights' in custom_dict and 'mixing_weights' in pretrained_dict and custom_dict['mixing_weights'].shape != pretrained_dict['mixing_weights'].shape:
            # Keep the first four weights from the pretrained model, and randomly initialize the fifth weight
            custom_dict['mixing_weights'][:4] = pretrained_dict['mixing_weights'][:4]
            custom_dict['mixing_weights'][4] = torch.zeros_like(custom_dict['mixing_weights'][4])
        else:
            custom_dict['mixing_weights'][:4] = pretrained_dict['mixing_weights'][:4]

        # Load the weights that do match
        matching_keys = {k: v for k, v in pretrained_dict.items() if k in custom_dict and k != 'mixing_weights'}
        custom_dict.update(matching_keys)

        # Now load the updated state_dict
        self.load_state_dict(custom_dict, strict=False)
        
    def forward(self, batch, pose=None):
        """
        Assumes batch is shape (B, C, H, W) where C is the concatentation of all layer features.
        """
        if self.feat_map_dropout > 0 and self.training:
            batch = F.dropout(batch, p=self.feat_map_dropout)
        
        output_feature = None
        start = 0
        mixing_weights = torch.nn.functional.softmax(self.mixing_weights, dim=0)
        if self.pos_embedding is not None: #position embedding
            batch = torch.cat((batch, self.pos_embedding), dim=1)
        for i in range(len(mixing_weights)):
            # Share bottleneck layers across timesteps
            bottleneck_layer = self.bottleneck_layers[i % len(self.feature_dims)]
            # Chunk the batch according the layer
            # Account for looping if there are multiple timesteps
            end = start + self.feature_dims[i % len(self.feature_dims)]
            feats = batch[:, start:end, :, :]
            start = end
            # Downsample the number of channels and weight the layer
            bottlenecked_feature = bottleneck_layer(feats)
            bottlenecked_feature = mixing_weights[i] * bottlenecked_feature
            if output_feature is None:
                output_feature = bottlenecked_feature
            else:
                output_feature += bottlenecked_feature
        
        if self.last_layer is not None:

            output_feature_after = self.last_layer(output_feature)
            if self.skip_connection:
                output_feature = output_feature + output_feature_after
        return output_feature
    

def model(pretrained: bool = False):
    device = 'cpu'
    aggre_net = AggregationNetwork(feature_dims=[768,], projection_dim=768, device=device, feat_map_dropout=0.2)
    if pretrained:
        ckpt_dir = 'ckpts/0300_dino_spair/best.pth'
        pretrained_dict = torch.load(ckpt_dir, map_location=device)
        aggre_net.load_pretrained_weights(pretrained_dict)
    return aggre_net