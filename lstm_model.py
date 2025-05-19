import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class LSTMTrafficModel(nn.Module):
    """
    LSTM model for traffic flow prediction
    Args:
        input_size: Number of features in the input (1 for univariate time series)
        hidden_size: Number of LSTM units in each layer
        num_layers: Number of LSTM layers
        output_size: Number of output features (1 for regression)
        dropout: Dropout probability (0 for no dropout)
    """
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMTrafficModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def train_lstm_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    """
    Train the LSTM model and evaluate on test set
    Returns:
        Tuple of (train_losses, test_losses, predictions, actuals)
    """
    model.to(device)
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # Average training loss for the epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluate on test set
        model.eval()
        test_loss = 0
        all_preds = []
        all_actuals = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                test_loss += criterion(outputs, batch_y).item()
                
                all_preds.extend(outputs.cpu().numpy())
                all_actuals.extend(batch_y.cpu().numpy())
        
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
    
    # Calculate metrics
    mse = mean_squared_error(all_actuals, all_preds)
    mae = mean_absolute_error(all_actuals, all_preds)
    rmse = np.sqrt(mse)
    print(f'\nFinal Metrics - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}')
    
    return train_losses, test_losses, np.array(all_preds), np.array(all_actuals)

def evaluate_lstm_model(model, test_loader, device='cpu'):
    """Evaluate LSTM model on test set and return predictions and metrics"""
    model.eval()
    all_preds = []
    all_actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            
            all_preds.extend(outputs.cpu().numpy())
            all_actuals.extend(batch_y.cpu().numpy())
    
    all_preds = np.array(all_preds).flatten()
    all_actuals = np.array(all_actuals).flatten()
    
    mse = mean_squared_error(all_actuals, all_preds)
    mae = mean_absolute_error(all_actuals, all_preds)
    rmse = np.sqrt(mse)
    
    print(f'Evaluation Metrics - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}')
    
    return all_preds, all_actuals

def plot_lstm_results(train_losses, test_losses, predictions=None, actuals=None):
    """Plot LSTM training/test losses and predictions vs actuals if provided"""
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LSTM Training and Test Loss')
    plt.legend()
    
    if predictions is not None and actuals is not None:
        # Plot predictions vs actuals
        plt.subplot(1, 2, 2)
        plt.scatter(actuals, predictions, alpha=0.3)
        plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('LSTM Predictions vs Actuals')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from data_preprocessing import get_data_loaders
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders("Scats_Data_October_2006.xlsx")
    
    if train_loader is not None and test_loader is not None:
        # Initialize LSTM model
        lstm_model = LSTMTrafficModel(input_size=1, hidden_size=64, num_layers=2)
        
        # Training parameters
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
        epochs = 10
        
        # Train and evaluate LSTM
        print("\nTraining LSTM Model...")
        train_losses, test_losses, preds, actuals = train_lstm_model(
            lstm_model, train_loader, test_loader, criterion, optimizer, epochs, device
        )
        plot_lstm_results(train_losses, test_losses, preds, actuals)

# import torch
# import torch.nn as nn
# import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import matplotlib.pyplot as plt
# import os

# class LSTMTrafficModel(nn.Module):
#     """Enhanced LSTM model with weight saving capability"""
#     def __init__(self, input_size=1, hidden_size=64, num_layers=2):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)
        
#     def forward(self, x):
#         out, _ = self.lstm(x)
#         return self.fc(out[:, -1, :])

# def train_and_save_model():
#     # Setup device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Initialize model
#     model = LSTMTrafficModel().to(device)
    
#     # Load data (make sure data_preprocessing.py is in same directory)
#     from data_preprocessing import get_data_loaders
#     train_loader, test_loader = get_data_loaders("Scats_Data_October_2006.xlsx")
    
#     if train_loader is None:
#         print("Data loading failed! Check your data path.")
#         return

#     # Training setup
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
#     print("\nTraining LSTM Model...")
#     for epoch in range(10):  # 10 epochs
#         model.train()
#         for batch_x, batch_y in train_loader:
#             batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#             optimizer.zero_grad()
#             outputs = model(batch_x)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()
        
#         # Print progress
#         if (epoch+1) % 2 == 0:
#             print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

#     # Save trained weights
#     os.makedirs('models', exist_ok=True)
#     torch.save(model.state_dict(), 'models/lstm_weights.pth')
#     print("\nTraining complete! Saved weights to models/lstm_weights.pth")
    
#     # Evaluation
#     model.eval()
#     preds, actuals = [], []
#     with torch.no_grad():
#         for batch_x, batch_y in test_loader:
#             outputs = model(batch_x.to(device))
#             preds.extend(outputs.cpu().numpy())
#             actuals.extend(batch_y.numpy())
    
#     # Calculate metrics
#     rmse = np.sqrt(mean_squared_error(actuals, preds))
#     print(f"\nFinal RMSE: {rmse:.4f}")

# if __name__ == "__main__":
#     train_and_save_model()
