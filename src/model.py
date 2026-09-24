import torch.nn as nn

class VoiceClassifier(nn.Module):
    def __init__(self):
        super(VoiceClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1300, 1024),
            nn.Linear(1024, 768),
            nn.Linear(768, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 64),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):           
        return self.network(x)
