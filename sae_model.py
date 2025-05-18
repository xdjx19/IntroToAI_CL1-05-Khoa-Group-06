import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
from collections import defaultdict

class SparseAutoencoder(nn.Module):
    """SAE with proper shape handling"""
    def __init__(self, input_dim=24, encoding_dim=12, hidden_dim=48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoding_dim)
        )  # Added missing closing parenthesis here
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim))
        
    def forward(self, x):
        original_shape = x.shape
        if len(original_shape) == 3:
            x = x.view(original_shape[0], -1)  # Flatten to [batch, seq_len*features]
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        if len(original_shape) == 3:
            decoded = decoded.view(original_shape)  # Restore original shape
        return decoded

def train_and_save_sae():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_length = 24  # Must match data_preprocessing.py
    
    model = SparseAutoencoder(input_dim=seq_length).to(device)
    from data_preprocessing import get_data_loaders
    train_loader, test_loader = get_data_loaders("Scats_Data_October_2006.xlsx")
    
    if train_loader is None:
        print("Data loading failed!")
        return

    # Debug: Print first batch shape
    sample_batch, _ = next(iter(train_loader))
    print(f"\nInput shape: {sample_batch.shape}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = defaultdict(list)
    
    print("Training SAE...")
    for epoch in range(20):
        model.train()
        epoch_loss = 0
        
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        if (epoch+1) % 5 == 0:
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/sae_weights.pth')
    print("\nTraining complete! Saved weights to models/sae_weights.pth")

    # Evaluation (FIXED: Flatten before MSE calculation)
    model.eval()
    reconstructions, originals = [], []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            outputs = model(batch_x.to(device))
            reconstructions.extend(outputs.cpu().numpy().reshape(-1))  # Flatten
            originals.extend(batch_x.numpy().reshape(-1))  # Flatten
    
    mse = mean_squared_error(originals, reconstructions)
    print(f"\nFinal Metrics:")
    print(f"- Reconstruction MSE: {mse:.4f}")
    print(f"- Total samples evaluated: {len(originals)}")

    # Visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(originals[:1000], reconstructions[:1000], alpha=0.3)  # Plot first 1000 points
    plt.plot([min(originals), max(originals)], [min(originals), max(originals)], 'r--')
    plt.xlabel('Original Values')
    plt.ylabel('Reconstructions')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.suptitle('SAE Performance')
    plt.tight_layout()
    plt.savefig('sae_results.png')
    plt.show()

if __name__ == "__main__":
    train_and_save_sae()
