from diy_sc.projection_network import agg_sd_dino, agg_sd_dino_in3d_spair, agg_sd_dino_in3d, agg_dino, agg_dino_in3d_spair, agg_dino_in3d, agg_dino_384, agg_dino_128

agg_net = agg_sd_dino(pretrained=True)
print(agg_net.num_params,agg_net)

agg_net = agg_sd_dino_in3d_spair(pretrained=True)
print(agg_net.num_params,agg_net)

agg_net = agg_sd_dino_in3d(pretrained=True)
print(agg_net.num_params,agg_net)


agg_net = agg_dino(pretrained=True)
print(agg_net.num_params,agg_net)

agg_net = agg_dino_in3d_spair(pretrained=True)
print(agg_net.num_params,agg_net)

agg_net = agg_dino_in3d(pretrained=True)
print(agg_net.num_params,agg_net)

agg_net = agg_dino_384(pretrained=True)
print(agg_net.num_params,agg_net)

agg_net = agg_dino_128(pretrained=True)
print(agg_net.num_params,agg_net)
