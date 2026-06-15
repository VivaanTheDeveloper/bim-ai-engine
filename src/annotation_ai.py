import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ArchitecturalPlacementModel(nn.Module):
    """
    Deep Convolutional Autoencoder designed to predict 
    optimal coordinate zones for text and dimension placement.
    """
    def __init__(self):
        super(ArchitecturalPlacementModel, self).__init__()
        
        # Encoder: Compresses raw un-cluttered walls geometry layout arrays
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), # [batch, 32, 128, 128]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [batch, 64, 64, 64]
            nn.ReLU(),
        )
        
        # Decoder: Generates a spatial coordinate heatmap mapping ideal dimension positions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # Squashes output pixels between 0.0 and 1.0 (Probability Map)
        )

    def forward(self, x):
        encoded_features = self.encoder(x)
        heatmap_prediction = self.decoder(encoded_features)
        return heatmap_prediction

from pathlib import Path

class RealBIMDataset(Dataset):
    """Loads real preprocessed IFC + DXF tensor pairs for training."""
    def __init__(self):
        self.x_files = sorted(Path("dataset/processed_x").glob("*.npy"))
        self.y_files = sorted(Path("dataset/processed_y").glob("*.npy"))
        assert len(self.x_files) == len(self.y_files), "Mismatched training pairs"
        print(f"Dataset loaded: {len(self.x_files)} real training samples found")

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):
        x = np.load(str(self.x_files[idx]))
        y = np.load(str(self.y_files[idx]))
        return torch.from_numpy(x), torch.from_numpy(y)

def run_ai_training_loop():
    dataset = RealBIMDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ArchitecturalPlacementModel().to(device)
    criterion = nn.BCELoss() # Binary Cross Entropy evaluates pixel classification overlays
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Booting Neural Blueprint Optimizer Engine on architecture target: {device}")
    
    for epoch in range(10):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Training Epoch {epoch+1}/10 | Target Layout Convergence Loss: {epoch_loss/len(dataloader):.6f}")
        
    torch.save(model.state_dict(), "models/drafting_ai.pth")
    print("Model optimization complete. Matrix configurations saved to models/drafting_ai.pth")

if __name__ == "__main__":
    run_ai_training_loop()