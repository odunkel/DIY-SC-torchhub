import torch

agg_net = torch.hub.load('odunkel/DIY-SC-torchhub', 'agg_dino', pretrained=True)
print(agg_net)
agg_net = torch.hub.load('odunkel/DIY-SC-torchhub', 'agg_sd_dino', pretrained=True)
agg_net = torch.hub.load('odunkel/DIY-SC-torchhub', 'agg_dino_384', pretrained=True)
agg_net = torch.hub.load('odunkel/DIY-SC-torchhub', 'agg_dino_128', pretrained=True)