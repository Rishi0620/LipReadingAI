import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalLipReader(nn.Module):
    def __init__(self, num_classes):
        super(MinimalLipReader, self).__init__()

        # 3D CNN: (B, 1, T, H, W) -> (B, 64, T//2, 28, 28)
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2))  # Reduce H, W
        )

        # Bidirectional LSTM input size is 64 * 28 * 28
        self.lstm = nn.LSTM(
            input_size=64 * 28 * 28,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # Final classification layer
        self.classifier = nn.Linear(256 * 2, num_classes)  # BiLSTM → 2x hidden

    def forward(self, x):
        # x: [B, T, C, H, W] → [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)

        x = self.cnn(x)  # [B, 64, T, H, W]
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        x = x.reshape(B, T, -1)  # [B, T, C*H*W]

        x, _ = self.lstm(x)  # [B, T, 2*hidden]
        x = self.classifier(x)  # [B, T, num_classes]
        x = x.log_softmax(dim=-1)  # For CTC Loss
        return x