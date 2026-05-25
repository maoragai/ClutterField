import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
if __name__ == "__main__":

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
        torch.nn.Sigmoid()
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = torch.nn.MSELoss()

    radius = 0.5

    for epoch in tqdm(range(1000)):

        # Uniform samples in [-1, 1]
        inputs = 2 * torch.rand(32, 2) - 1

        # Circle occupancy labels
        targets = (
            torch.sum(inputs ** 2, dim=1, keepdim=True)
            < radius ** 2
        ).float()

        # Forward pass
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")


    resolution = 200
    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)

    xx, yy = torch.meshgrid(x, y, indexing='xy')

    grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    # Run model
    with torch.no_grad():
        predictions = model(grid)

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

    plt.title('Learned Occupancy Field')

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()