import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
if __name__ == "__main__":

    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
        torch.nn.Sigmoid()
    )

    radius = 0.5
    criterion = torch.nn.MSELoss()
    coarse_grid_res=32
    mid_grid_res = coarse_grid_res * 2 
    fine_grid_res = coarse_grid_res  * 4
    ultra_grid_res = coarse_grid_res    * 8
    feature_dim=4
    
    # grid_shape = (mid_grid_res, mid_grid_res, feature_dim)
    def spatial_hash(coords:torch.Tensor, 
                     table_size:torch.Tensor):
        x = coords[:, 0].long()
        y = coords[:, 1].long()

        # Simple hash: (x * prime1 + y * prime2) % table_size
        prime1 = 73856093
        prime2 = 19349663
        table_size = table_size 

        indices = (x * prime1 + y * prime2) % table_size

        return indices
    
    class GridEncoder(nn.Module):
        def __init__(self, grid_shape):
            super().__init__()
            self.resolution = grid_shape[0]
            self.table_size = grid_shape[1]
            self.feature_dim = grid_shape[2]
            self.grid_shape = grid_shape
            self.hash_table = nn.Parameter(torch.randn(*grid_shape[-2:]) * 0.01)
        
        def forward(self, coords):

            # [-1, 1] -> [0, grid_size -1]
            coords = (coords + 1) / 2
            coords = coords * (self.resolution - 1)
            x = coords[:, 0]
            y = coords[:, 1]

            # Integer corner coordinates
            x0 = torch.floor(x).long()
            x1 = x0 + 1

            y0 = torch.floor(y).long()
            y1 = y0 + 1

            # Clamp bounds
            x0 = torch.clamp(x0, 0, self.resolution - 1)
            x1 = torch.clamp(x1, 0, self.resolution - 1)

            y0 = torch.clamp(y0, 0, self.resolution - 1)
            y1 = torch.clamp(y1, 0, self.resolution - 1)
            # Retrieve corner features
            id00 = spatial_hash(torch.stack([x0.float(), y0.float()], dim=-1), self.table_size)  
            id10 = spatial_hash(torch.stack([x1.float(), y0.float()], dim=-1), self.table_size)
            id01 = spatial_hash(torch.stack([x0.float(), y1.float()], dim=-1), self.table_size)
            id11 = spatial_hash(torch.stack([x1.float(), y1.float()], dim=-1), self.table_size)
            
            f00 = self.hash_table[id00]
            f10 = self.hash_table[id10]
            f01 = self.hash_table[id01]
            f11 = self.hash_table[id11]
            # Fractional offsets
            wx = (x - x0.float()).unsqueeze(-1)
            wy = (y - y0.float()).unsqueeze(-1)

            # Interpolate along x
            fx0 = f00 * (1 - wx) + f10 * wx
            fx1 = f01 * (1 - wx) + f11 * wx

            # Interpolate along y
            features = fx0 * (1 - wy) + fx1 * wy

            return features
    
    
    class MultiResolutionHashGridEncoder(nn.Module):

            def __init__(self,
                        resolutions:list[int],
                        table_size:int,
                        feature_dim):

                super().__init__()

                self.encoders = nn.ModuleList()

                for resolution in resolutions:

                    grid_shape = (
                        resolution,
                        table_size,
                        feature_dim
                    )

                    self.encoders.append(
                        GridEncoder(grid_shape)
                    )

            def forward(self, coords):

                features = []

                for encoder in self.encoders:

                    feat = encoder(coords)

                    features.append(feat)

                return torch.cat(features, dim=-1)
    gridencoder = MultiResolutionHashGridEncoder(
        resolutions=[coarse_grid_res, mid_grid_res, fine_grid_res, ultra_grid_res],
        table_size=2048,
        feature_dim=feature_dim
        )

    optimizer = torch.optim.Adam(
        list(model.parameters()) +
        list(gridencoder.parameters()),
        lr=1e-3
    )
    for epoch in tqdm(range(1000)):
        inputs = torch.rand(1024, 2) * 2 - 1  # Random points in [-1, 1]
        # Circle occupancy labels
        freq = 10.0

        targets = (
            (
                torch.sin(freq * inputs[:,0]) *
                torch.sin(freq * inputs[:,1])
            ) > 0
        ).float().unsqueeze(-1)

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