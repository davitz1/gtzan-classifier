import torch
import torch.nn as nn

class MFCC_CNN(nn.Module):
    def __init__(self, num_classes=10, input_shape=(1, 130, 13)):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
    )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(32)
        )

        dummy = torch.zeros(1, *input_shape)
        with torch.no_grad():
            out = self._forward_features(dummy)
        flattened_dim = out.numel()

        self.fc = nn.Sequential(
            nn.Linear(flattened_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, num_classes)
        )

    def _forward_features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def summary(self, input_shape):
        """Helper to print model summary and check shapes"""
        x = torch.randn(1, *input_shape)
        print(f"Input shape: {x.shape}")
        x = self.block1(x); print(f"After Block1: {x.shape}")
        x = self.block2(x); print(f"After Block2: {x.shape}")
        x = self.block3(x); print(f"After Block3: {x.shape}")
        x = x.flatten(1); print(f"After Flatten: {x.shape}")
        x = self.fc(x); print(f"After FC: {x.shape}")

if __name__ == "__main__":
    model = MFCC_CNN()
    model.summary((1, 130, 13))