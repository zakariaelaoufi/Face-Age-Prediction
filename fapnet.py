import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FAPNet(nn.Module):
    def __init__(self, output_dim=1):
        super(FAPNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(256 * 12 * 12, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)

        self.output = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        return self.output(x)


def face_age_prediction(model, image):
    # Si enhaced est un numpy array, convertis-le
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    # print("Original shape:", enhaced.shape)

    # Squeeze all dimensions of size 1 to remove extra dimensions
    image = image.squeeze()
    # print("After squeeze:", enhaced.shape)

    # Vérifie que le type est float
    if image.dtype != torch.float32:
        image = image.float()

    # Normalise si nécessaire (si les valeurs sont dans [0,255])
    if image.max() > 1.0:
        image = image / 255.0

    # Si nécessaire, permute les dimensions de (H, W, C) à (C, H, W)
    if image.ndim == 3 and image.shape[2] == 3:
        image = image.permute(2, 0, 1)
        # print("After permute (C, H, W):", enhaced.shape)

    # Ajoute UNE SEULE dimension batch pour obtenir (N, C, H, W)
    if image.ndim == 3:  # Si on a (C, H, W)
        image = image.unsqueeze(0)  # Devient (1, C, H, W)

    # print("Final shape before model:", enhaced.shape)  # Doit être [1, 3, H, W]

    # Passe au modèle
    model.eval()
    with torch.no_grad():
        return model(image).squeeze().item()