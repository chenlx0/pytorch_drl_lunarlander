import torch.nn as nn

def weight_init(net : nn.Module):
    for module in net.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            # nn.init.kaiming_uniform_(module.weight, mode='fan_out', nonlinearity='relu')

            if module.bias is not None:
                nn.init.zeros_(module.bias)

class DRLNet(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DRLNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x

class PGNet(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DRLNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(),
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x
