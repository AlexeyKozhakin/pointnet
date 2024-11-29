import torch
import torch.nn as nn

class PointNet(nn.Module):
    def __init__(self, num_classes):
        super(PointNet, self).__init__()
        self.mlp1 = nn.Sequential(nn.Conv1d(6, 64, 1), nn.ReLU(), nn.BatchNorm1d(64))
        self.mlp2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.ReLU(), nn.BatchNorm1d(128))
        self.mlp3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.ReLU(), nn.BatchNorm1d(1024))
        self.mlp4 = nn.Sequential(nn.Conv1d(1024, 512, 1), nn.ReLU(), nn.BatchNorm1d(512))
        self.mlp5 = nn.Sequential(nn.Conv1d(512, 256, 1), nn.ReLU(), nn.BatchNorm1d(256))
        self.fc = nn.Conv1d(256, num_classes, 1)  # Выходной слой

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, C, N]
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        x = self.mlp5(x)
        x = self.fc(x)  # [B, num_classes, N]
        x = x.permute(0, 2, 1)  # [B, N, num_classes]
        return x
