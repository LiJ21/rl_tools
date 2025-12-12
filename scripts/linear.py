import torch
import torch.nn as nn
import sys

class LinearQNet(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(feature_dim, action_dim, bias=False)

    def forward(self, x):
        return self.fc(x)

FeatureDim = int(sys.argv[1])
ActionDim = int(sys.argv[2])

model = LinearQNet(FeatureDim, ActionDim)
print(model)
example = torch.randn(1, FeatureDim)
scripted = torch.jit.trace(model, example)
scripted.save("qnet.pt")

