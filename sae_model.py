import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from collections import defaultdict

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder (SAE) model with L1 regularization for sparsity
    Args:
        input_dim: Dimension of input features
        encoding_dim: Dimension of the bottleneck layer (encoding)
        hidden_dims: List of dimensions for hidden layers (decoder/encoder)
        activation: Activation function to use ('relu', 'sigmoid', 'tanh')
        sparsity_weight: Weight for the sparsity regularization term
        sparsity_target: Desired average activation of hidden units
    """
    def __init__(self, input_dim=64, encoding_dim=32, hidden_dims=[128], 
                 activation='relu', sparsity_weight=0.05, sparsity_target=0.1):
        super(SparseAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.sparsity_weight = sparsity_weight
        self.sparsity_target = sparsity_target
        
        # Determine activation function
        self.activation = {
            'relu': F.relu,
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh
        }.get(activation, F.relu)
        
        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.ModuleList(encoder_layers)
        
        # Decoder layers
        decoder_layers = []
        prev_dim = encoding_dim
        for dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.ModuleList(decoder_layers)
        
    def forward(self, x):
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
            
        # Encoding
        h = x
        for i, layer in enumerate(self.encoder):
            h = layer(h)
            if i < len(self.encoder) - 1:  # No activation after last encoder layer
                h = self.activation(h)
        encoding = h
        
        # Track activations for sparsity regularization
        if self.training:
            self.activations = encoding.detach().mean(dim=0)
        
        # Decoding
        for i, layer in enumerate(self.decoder):
            encoding = layer(encoding)
            if i < len(self.decoder) - 1:  # No activation after last decoder layer
                encoding = self.activation(encoding)
        
        return encoding
    
    def get_sparsity_loss(self):
        """Calculate KL divergence sparsity loss"""
        if not hasattr(self, 'activations'):
            return torch.tensor(0.0)
            
        # KL divergence between target sparsity and actual activations
        sparsity = self.activations.mean()
        kl_div = self.sparsity_target * torch.log(self.sparsity_target / sparsity) + \
                 (1 - self.sparsity_target) * torch.log((1 - self.sparsity_target) / (1 - sparsity))
        return kl_div

def train_sae_model(model, train_loader, test_loader, criterion, optimizer, 
                   num_epochs=10, device='cpu', verbose=True):
    """
    Train the Sparse Autoencoder model
    Returns:
        Dictionary containing training history and results
    """
    model.to(device)
    history = defaultdict(list)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_sparsity_loss = 0
        
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]  # Assume first element is input data
                
            batch = batch.to(device)
            
            # Forward pass
            outputs = model(batch)
            
            # Calculate losses
            recon_loss = criterion(outputs, batch)
            sparsity_loss = model.get_sparsity_loss()
            total_loss = recon_loss + model.sparsity_weight * sparsity_loss
            
            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            epoch_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
        
        # Average losses for the epoch
        avg_loss = epoch_loss / len(train_loader)
        avg_recon = epoch_recon_loss / len(train_loader)
        avg_sparsity = epoch_sparsity_loss / len(train_loader)
        
        history['train_loss'].append(avg_loss)
        history['train_recon'].append(avg_recon)
        history['train_sparsity'].append(avg_sparsity)
        
        # Evaluate on test set
        test_metrics = evaluate_sae_model(model, test_loader, criterion, device)
        for key, val in test_metrics.items():
            history[f'test_{key}'].append(val)
        
        if verbose:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, Sparsity: {avg_sparsity:.4f}), '
                  f'Test Loss: {test_metrics["loss"]:.4f}')

    return history

def evaluate_sae_model(model, test_loader, criterion, device='cpu'):
    """Evaluate SAE model on test set and return metrics"""
    model.eval()
    metrics = defaultdict(float)
    all_outputs = []
    all_inputs = []
    
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]  # Assume first element is input data
                
            batch = batch.to(device)
            outputs = model(batch)
            
            # Calculate losses
            recon_loss = criterion(outputs, batch)
            sparsity_loss = model.get_sparsity_loss()
            total_loss = recon_loss + model.sparsity_weight * sparsity_loss
            
            # Accumulate metrics
            metrics['loss'] += total_loss.item()
            metrics['recon'] += recon_loss.item()
            metrics['sparsity'] += sparsity_loss.item()
            
            # Store for reconstruction metrics
            all_outputs.append(outputs.cpu())
            all_inputs.append(batch.cpu())
    
    # Average metrics
    for key in metrics:
        metrics[key] /= len(test_loader)
    
    # Calculate reconstruction metrics
    all_outputs = torch.cat(all_outputs).numpy()
    all_inputs = torch.cat(all_inputs).numpy()
    metrics['mse'] = mean_squared_error(all_inputs, all_outputs)
    
    return metrics

def plot_sae_results(history, sample_inputs=None, sample_reconstructions=None):
    """Plot SAE training results and sample reconstructions"""
    plt.figure(figsize=(15, 5))
    
    # Plot training history
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Total Loss')
    plt.plot(history['train_recon'], label='Train Recon Loss')
    plt.plot(history['test_loss'], label='Test Total Loss')
    plt.plot(history['test_recon'], label='Test Recon Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    
    # Plot sparsity
    plt.subplot(1, 3, 2)
    plt.plot(history['train_sparsity'], label='Train Sparsity')
    plt.plot(history['test_sparsity'], label='Test Sparsity')
    plt.xlabel('Epoch')
    plt.ylabel('Sparsity Loss')
    plt.title('Sparsity Training')
    plt.legend()
    
    # Plot sample reconstructions if provided
    if sample_inputs is not None and sample_reconstructions is not None:
        plt.subplot(1, 3, 3)
        plt.scatter(sample_inputs.flatten(), sample_reconstructions.flatten(), alpha=0.3)
        plt.plot([sample_inputs.min(), sample_inputs.max()], 
                 [sample_inputs.min(), sample_inputs.max()], 'r--')
        plt.xlabel('Original Values')
        plt.ylabel('Reconstructed Values')
        plt.title('Reconstruction Quality')
    
    plt.tight_layout()
    plt.show()

def visualize_encodings(model, data_loader, device='cpu', max_samples=1000):
    """Visualize the encoded representations of the data"""
    model.eval()
    encodings = []
    labels = []  # If available
    
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
                if len(batch) > 1:
                    labels.extend(batch[1].cpu().numpy())
            else:
                inputs = batch
                
            inputs = inputs.to(device)
            
            # Get encodings
            x = inputs
            for i, layer in enumerate(model.encoder):
                x = layer(x)
                if i < len(model.encoder) - 1:
                    x = model.activation(x)
            encodings.append(x.cpu().numpy())
            
            if len(encodings) * encodings[0].shape[0] >= max_samples:
                break
    
    encodings = np.concatenate(encodings)[:max_samples]
    
    plt.figure(figsize=(10, 6))
    if len(labels) >= len(encodings):
        plt.scatter(encodings[:, 0], encodings[:, 1], c=labels[:len(encodings)], alpha=0.6)
        plt.colorbar()
    else:
        plt.scatter(encodings[:, 0], encodings[:, 1], alpha=0.6)
    plt.title('2D Visualization of Encodings')
    plt.xlabel('Encoding Dimension 1')
    plt.ylabel('Encoding Dimension 2')
    plt.show()

if __name__ == "__main__":
    # Example usage with synthetic data
    from torch.utils.data import DataLoader, TensorDataset
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create synthetic data
    torch.manual_seed(42)
    data = torch.randn(1000, 64)  # 1000 samples with 64 features
    train_data = data[:800]
    test_data = data[800:]
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # Initialize SAE model
    sae_model = SparseAutoencoder(
        input_dim=64,
        encoding_dim=16,
        hidden_dims=[32],
        activation='relu',
        sparsity_weight=0.1,
        sparsity_target=0.05
    )
    
    # Training parameters
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(sae_model.parameters(), lr=0.001)
    epochs = 20
    
    # Train and evaluate SAE
    print("\nTraining Sparse Autoencoder...")
    history = train_sae_model(
        sae_model, train_loader, test_loader, criterion, optimizer, epochs, device
    )
    
    # Get some sample reconstructions
    sample_inputs = test_data[:100].to(device)
    with torch.no_grad():
        sample_reconstructions = sae_model(sample_inputs).cpu().numpy()
    
    # Plot results
    plot_sae_results(history, test_data[:100].numpy(), sample_reconstructions)
    
    # Visualize encodings
    visualize_encodings(sae_model, test_loader, device)

# import torch
# import torch.nn as nn
# import numpy as np
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt
# import os
# from collections import defaultdict

# class SparseAutoencoder(nn.Module):
#     """SAE with proper shape handling"""
#     def __init__(self, input_dim=24, encoding_dim=12, hidden_dim=48):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, encoding_dim)
#         )  # Added missing closing parenthesis here
#         self.decoder = nn.Sequential(
#             nn.Linear(encoding_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, input_dim))
        
#     def forward(self, x):
#         original_shape = x.shape
#         if len(original_shape) == 3:
#             x = x.view(original_shape[0], -1)  # Flatten to [batch, seq_len*features]
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         if len(original_shape) == 3:
#             decoded = decoded.view(original_shape)  # Restore original shape
#         return decoded

# def train_and_save_sae():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     seq_length = 24  # Must match data_preprocessing.py
    
#     model = SparseAutoencoder(input_dim=seq_length).to(device)
#     from data_preprocessing import get_data_loaders
#     train_loader, test_loader = get_data_loaders("Scats_Data_October_2006.xlsx")
    
#     if train_loader is None:
#         print("Data loading failed!")
#         return

#     # Debug: Print first batch shape
#     sample_batch, _ = next(iter(train_loader))
#     print(f"\nInput shape: {sample_batch.shape}")

#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     history = defaultdict(list)
    
#     print("Training SAE...")
#     for epoch in range(20):
#         model.train()
#         epoch_loss = 0
        
#         for batch_x, _ in train_loader:
#             batch_x = batch_x.to(device)
#             outputs = model(batch_x)
#             loss = criterion(outputs, batch_x)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
        
#         avg_loss = epoch_loss / len(train_loader)
#         history['train_loss'].append(avg_loss)
#         if (epoch+1) % 5 == 0:
#             print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

#     # Save model
#     os.makedirs('models', exist_ok=True)
#     torch.save(model.state_dict(), 'models/sae_weights.pth')
#     print("\nTraining complete! Saved weights to models/sae_weights.pth")

#     # Evaluation (FIXED: Flatten before MSE calculation)
#     model.eval()
#     reconstructions, originals = [], []
#     with torch.no_grad():
#         for batch_x, _ in test_loader:
#             outputs = model(batch_x.to(device))
#             reconstructions.extend(outputs.cpu().numpy().reshape(-1))  # Flatten
#             originals.extend(batch_x.numpy().reshape(-1))  # Flatten
    
#     mse = mean_squared_error(originals, reconstructions)
#     print(f"\nFinal Metrics:")
#     print(f"- Reconstruction MSE: {mse:.4f}")
#     print(f"- Total samples evaluated: {len(originals)}")

#     # Visualization
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.scatter(originals[:1000], reconstructions[:1000], alpha=0.3)  # Plot first 1000 points
#     plt.plot([min(originals), max(originals)], [min(originals), max(originals)], 'r--')
#     plt.xlabel('Original Values')
#     plt.ylabel('Reconstructions')
    
#     plt.subplot(1, 2, 2)
#     plt.plot(history['train_loss'])
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
    
#     plt.suptitle('SAE Performance')
#     plt.tight_layout()
#     plt.savefig('sae_results.png')
#     plt.show()

# if __name__ == "__main__":
#     train_and_save_sae()
