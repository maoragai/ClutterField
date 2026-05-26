import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
if __name__ == "__main__":

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
        torch.nn.Sigmoid()
    )

    radius = 0.5
    criterion = torch.nn.MSELoss()
    grid_res=32
    feature_dim=4
    grid_shape = (grid_res, grid_res,feature_dim)
    class GridEncoder(nn.Module):
        def __init__(self, grid_shape):
            super().__init__()
            self.grid_shape = grid_shape
            self.grid =nn.Parameter(torch.randn(*grid_shape)*0.01)
        def forward(self, coords):
            # coords: (N, 2) in [-1, 1]
            # Map to [0, grid_res-1]
            coords = (coords + 1) / 2 * (self.grid_shape[0] - 1)
            x_idx = torch.clamp(coords[:, 0].long(), 0, self.grid_shape[0] - 1)
            y_idx = torch.clamp(coords[:, 1].long(), 0, self.grid_shape[1] - 1)
            features = self.grid[x_idx, y_idx]  # (N, feature_dim)
            return features
    
    gridencoder = GridEncoder(grid_shape)
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) +
        list(gridencoder.parameters()),
        lr=1e-3
    )
    for epoch in tqdm(range(1000)):
        inputs = torch.rand(1024, 2) * 2 - 1  # Random points in [-1, 1]
        # Circle occupancy labels
        targets = (
            torch.sum(inputs ** 2, dim=1, keepdim=True)
            < radius ** 2
        ).float()

        # Forward pass
        outputs = model(gridencoder(inputs))

        loss = criterion(outputs, targets)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")


    # ==========================================
    # Visualization
    # ==========================================

    resolution = 200

    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)

    xx, yy = torch.meshgrid(x, y, indexing='xy')

    # Create coordinate grid
    coords = torch.stack(
        [xx.flatten(), yy.flatten()],
        dim=-1
    )

    # Run through encoder + model
    with torch.no_grad():

        features = gridencoder(coords)

        predictions = model(features)

    # Reshape into image
    image = predictions.reshape(resolution, resolution)

    # Plot
    plt.figure(figsize=(6,6))

    plt.imshow(
        image.numpy(),
        extent=[-1,1,-1,1],
        origin='lower',
        cmap='viridis'
    )

    plt.colorbar(label='Occupancy Probability')

    plt.title('Grid-Encoded Occupancy Field')

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()