import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=64, encoding_dim=32, hidden_dims=[128], 
                 activation='relu', sparsity_weight=0.05, sparsity_target=0.1):
        super(SparseAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.sparsity_weight = sparsity_weight
        self.sparsity_target = sparsity_target

        activations = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh
        }
        self.activation_cls = activations.get(activation, nn.ReLU)
        self.activation = self.activation_cls()

        # Build encoder and decoder
        self.encoder = self.build_mlp([input_dim] + hidden_dims + [encoding_dim], self.activation_cls)
        self.decoder = self.build_mlp([encoding_dim] + list(reversed(hidden_dims)) + [input_dim], self.activation_cls)

        self._register_hooks()

    def build_mlp(self, layers, activation):
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(activation())
        return nn.Sequential(*modules)

    def _register_hooks(self):
        def hook(module, input, output):
            self.activations = output.mean(dim=0)
        self.encoder[-1].register_forward_hook(hook)

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        encoding = self.encoder(x)
        reconstruction = self.decoder(encoding)
        return reconstruction

    def get_sparsity_loss(self):
        if not hasattr(self, 'activations'):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        sparsity = torch.clamp(self.activations, 1e-6, 1 - 1e-6)
        target = torch.tensor(self.sparsity_target, device=sparsity.device)
        kl_div = target * torch.log(target / sparsity) + \
                 (1 - target) * torch.log((1 - target) / (1 - sparsity))
        return kl_div.mean()

def train_sae_model(model, train_loader, test_loader, criterion, optimizer, 
                   num_epochs=10, device='cpu', verbose=True):
    model.to(device)
    history = defaultdict(list)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_sparsity_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = batch[0].to(device, non_blocking=True)

            outputs = model(inputs)
            recon_loss = criterion(outputs, inputs)
            sparsity_loss = model.get_sparsity_loss()
            total_loss = recon_loss + model.sparsity_weight * sparsity_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()

        avg_loss = epoch_loss / len(train_loader)
        avg_recon = epoch_recon_loss / len(train_loader)
        avg_sparsity = epoch_sparsity_loss / len(train_loader)

        history['train_loss'].append(avg_loss)
        history['train_recon'].append(avg_recon)
        history['train_sparsity'].append(avg_sparsity)

        test_metrics = evaluate_sae_model(model, test_loader, criterion, device)
        for key, val in test_metrics.items():
            history[f'test_{key}'].append(val)

        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, Sparsity: {avg_sparsity:.4f}), "
                  f"Test Loss: {test_metrics['loss']:.4f}")

    return history

def evaluate_sae_model(model, test_loader, criterion, device='cpu'):
    model.eval()
    metrics = defaultdict(float)
    all_outputs = []
    all_inputs = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(device, non_blocking=True)
            outputs = model(inputs)

            recon_loss = criterion(outputs, inputs)
            sparsity_loss = model.get_sparsity_loss()
            total_loss = recon_loss + model.sparsity_weight * sparsity_loss

            metrics['loss'] += total_loss.item()
            metrics['recon'] += recon_loss.item()
            metrics['sparsity'] += sparsity_loss.item()

            all_outputs.append(outputs.cpu())
            all_inputs.append(inputs.cpu())

    for key in metrics:
        metrics[key] /= len(test_loader)

    all_outputs = torch.cat(all_outputs).numpy()
    all_inputs = torch.cat(all_inputs).numpy()
    metrics['mse'] = mean_squared_error(all_inputs, all_outputs)

    return metrics

def plot_sae_results(history, sample_inputs=None, sample_reconstructions=None):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Total Loss')
    plt.plot(history['train_recon'], label='Train Recon Loss')
    plt.plot(history['test_loss'], label='Test Total Loss')
    plt.plot(history['test_recon'], label='Test Recon Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history['train_sparsity'], label='Train Sparsity')
    plt.plot(history['test_sparsity'], label='Test Sparsity')
    plt.xlabel('Epoch')
    plt.ylabel('Sparsity Loss')
    plt.title('Sparsity Training')
    plt.legend()

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
    model.eval()
    encodings = []
    labels = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0].to(device, non_blocking=True)

            x = inputs
            for i, layer in enumerate(model.encoder):
                x = layer(x)
                if i < len(model.encoder) - 1:
                    x = model.activation(x)

            encodings.append(x.cpu().numpy())
            if len(encodings) * x.shape[0] >= max_samples:
                break

    encodings = np.concatenate(encodings)[:max_samples]

    plt.figure(figsize=(10, 6))
    plt.scatter(encodings[:, 0], encodings[:, 1], alpha=0.6)
    plt.title('2D Visualization of Encodings')
    plt.xlabel('Encoding Dimension 1')
    plt.ylabel('Encoding Dimension 2')
    plt.show()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(42)
    data = torch.randn(1000, 64)
    train_data = data[:800]
    test_data = data[800:]

    train_loader = DataLoader(TensorDataset(train_data), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=32, shuffle=False)

    sae_model = SparseAutoencoder(
        input_dim=64,
        encoding_dim=16,
        hidden_dims=[32],
        activation='relu',
        sparsity_weight=0.1,
        sparsity_target=0.05
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(sae_model.parameters(), lr=0.001)
    epochs = 20

    print("\nTraining Sparse Autoencoder...")
    history = train_sae_model(sae_model, train_loader, test_loader, criterion, optimizer, epochs, device)

    sample_inputs = test_data[:100].to(device)
    with torch.no_grad():
        sample_reconstructions = sae_model(sample_inputs).cpu().numpy()

    plot_sae_results(history, test_data[:100].numpy(), sample_reconstructions)
    visualize_encodings(sae_model, test_loader, device)





#third






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



#streamlit or tinkter for gui view