import torch

model = torch.hub.load('odunkel/DIY-SC-torchhub', 'model', pretrained=True)

print(model)