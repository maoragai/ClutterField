import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# Gaussian Target
# ============================================================

def gaussian_2d(xx, yy, x0, y0, sigma=0.05):

    return torch.exp(
        -((xx - x0) ** 2 + (yy - y0) ** 2)
        / (2 * sigma**2)
    )


# ============================================================
# Hash Grid Encoder
# ============================================================

class HashGridEncoder(nn.Module):

    def __init__(self,
                 resolution,
                 table_size,
                 feature_dim):

        super().__init__()

        self.resolution = resolution
        self.table_size = table_size
        self.feature_dim = feature_dim

        self.hash_table = nn.Parameter(
            torch.randn(table_size, feature_dim) * 0.01
        )

    def spatial_hash(self, x, y):

        return (
            (x * 73856093) ^
            (y * 19349663)
        ) % self.table_size

    def forward(self, coords):

        # [-1,1] -> [0,resolution-1]
        coords = (coords + 1) / 2
        coords = coords * (self.resolution - 1)

        x = coords[:, 0]
        y = coords[:, 1]

        x0 = torch.floor(x).long()
        x1 = x0 + 1

        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, self.resolution - 1)
        x1 = torch.clamp(x1, 0, self.resolution - 1)

        y0 = torch.clamp(y0, 0, self.resolution - 1)
        y1 = torch.clamp(y1, 0, self.resolution - 1)

        h00 = self.spatial_hash(x0, y0)
        h10 = self.spatial_hash(x1, y0)

        h01 = self.spatial_hash(x0, y1)
        h11 = self.spatial_hash(x1, y1)

        f00 = self.hash_table[h00]
        f10 = self.hash_table[h10]

        f01 = self.hash_table[h01]
        f11 = self.hash_table[h11]

        wx = (x - x0.float()).unsqueeze(-1)
        wy = (y - y0.float()).unsqueeze(-1)

        fx0 = f00 * (1 - wx) + f10 * wx
        fx1 = f01 * (1 - wx) + f11 * wx

        features = fx0 * (1 - wy) + fx1 * wy

        return features


# ============================================================
# Multi Resolution Encoder
# ============================================================

class MultiResolutionHashEncoder(nn.Module):

    def __init__(self,
                 resolutions,
                 table_size,
                 feature_dim):

        super().__init__()

        self.encoders = nn.ModuleList()

        for resolution in resolutions:

            self.encoders.append(
                HashGridEncoder(
                    resolution=resolution,
                    table_size=table_size,
                    feature_dim=feature_dim
                )
            )

    def forward(self, coords):

        features = []

        for encoder in self.encoders:

            features.append(
                encoder(coords)
            )

        return torch.cat(features, dim=-1)


# ============================================================
# ClutterField
# ============================================================

class ClutterField(nn.Module):

    def __init__(self,
                 resolutions,
                 table_size,
                 feature_dim):

        super().__init__()

        self.encoder = MultiResolutionHashEncoder(
            resolutions=resolutions,
            table_size=table_size,
            feature_dim=feature_dim
        )

        total_features = len(resolutions) * feature_dim

        self.decoder = nn.Sequential(
            nn.Linear(total_features, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, coords):

        features = self.encoder(coords)

        return self.decoder(features)


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':

    device = 'cpu'

    # ========================================================
    # Spatial Grid
    # ========================================================

    resolution = 128

    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)

    xx, yy = torch.meshgrid(x, y, indexing='xy')

    coords = torch.stack(
        [xx.flatten(), yy.flatten()],
        dim=-1
    ).to(device)

    # ========================================================
    # Static Background Clutter
    # ========================================================

    background = (
        0.3 * torch.sin(4 * xx)
        +
        0.2 * torch.cos(3 * yy)
    )

    background = background.unsqueeze(-1)

    # ========================================================
    # Model
    # ========================================================

    model = ClutterField(
        resolutions=[16, 32, 64, 128],
        table_size=512,
        feature_dim=4
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    criterion = nn.MSELoss()

    # ========================================================
    # Persistent Memory
    # ========================================================

    persistent_scene = torch.zeros_like(background)

    innovation_memory = torch.zeros_like(background)

    scene_decay = 0.995
    scene_update = 0.005

    innovation_decay = 0.99
    innovation_update = 0.01

    # ========================================================
    # Training
    # ========================================================

    num_steps = 1000

    for step in tqdm(range(num_steps)):

        # ====================================================
        # Weak Moving Target
        # ====================================================

        t = step / 150.0

        target_x = 0.5 * math.sin(t)
        target_y = 0.5 * math.cos(t)

        target = gaussian_2d(
            xx,
            yy,
            target_x,
            target_y,
            sigma=0.04
        )

        # weak target
        target = 0.15 * target.unsqueeze(-1)

        # ====================================================
        # Noise
        # ====================================================

        noise = 0.02 * torch.randn_like(background)

        # ====================================================
        # Observation
        # ====================================================

        observation = background + target + noise

        # ====================================================
        # Persistent Scene Update
        # ====================================================

        persistent_scene = (
            scene_decay * persistent_scene
            +
            scene_update * observation
        )

        # ====================================================
        # Train Predictor
        # ====================================================

        target_scene = persistent_scene.reshape(-1, 1).to(device)

        prediction = model(coords)

        loss = criterion(prediction, target_scene)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # ====================================================
        # Innovation Residual
        # ====================================================

        prediction_image = prediction.reshape(
            resolution,
            resolution
        )

        residual = (
            observation.squeeze(-1)
            -
            prediction_image
        )

        # ====================================================
        # Innovation Memory
        # ====================================================

        innovation_memory = (
            innovation_decay * innovation_memory
            +
            innovation_update * torch.abs(
                residual.unsqueeze(-1)
            )
        )

        if step % 100 == 0:

            print(
                f'Step {step} | '
                f'Loss: {loss.item():.6f}'
            )

    # ========================================================
    # Final Visualization
    # ========================================================

    final_observation = observation.squeeze(-1)

    final_prediction = prediction_image

    final_residual = residual

    final_innovation = innovation_memory.squeeze(-1)

    # ========================================================
    # Plot All
    # ========================================================

    plt.figure(figsize=(20, 5))

    # ========================================================
    # Observation
    # ========================================================

    plt.subplot(1, 4, 1)

    plt.imshow(
        final_observation.numpy(),
        extent=[-1, 1, -1, 1],
        origin='lower',
        cmap='viridis'
    )

    plt.title('Radar Observation')

    plt.colorbar()

    # ========================================================
    # Prediction
    # ========================================================

    plt.subplot(1, 4, 2)

    plt.imshow(
        final_prediction.detach().numpy(),
        extent=[-1, 1, -1, 1],
        origin='lower',
        cmap='viridis'
    )

    plt.title('Predicted Scene')

    plt.colorbar()

    # ========================================================
    # Residual
    # ========================================================

    plt.subplot(1, 4, 3)

    plt.imshow(
        final_residual.detach().numpy(),
        extent=[-1, 1, -1, 1],
        origin='lower',
        cmap='inferno'
    )

    plt.title('Innovation Residual')

    plt.colorbar()

    # ========================================================
    # Innovation Memory
    # ========================================================

    plt.subplot(1, 4, 4)

    plt.imshow(
        final_innovation.detach().numpy(),
        extent=[-1, 1, -1, 1],
        origin='lower',
        cmap='inferno'
    )

    plt.title('Accumulated Innovation')

    plt.colorbar()

    plt.tight_layout()

    plt.show()