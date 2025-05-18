import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

class GRUTrafficModel(nn.Module):
    """Enhanced GRU model with standardized architecture"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def train_and_save_gru():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = GRUTrafficModel().to(device)
    
    # Load data
    from data_preprocessing import get_data_loaders
    train_loader, test_loader = get_data_loaders("Scats_Data_October_2006.xlsx")
    
    if train_loader is None:
        print("Data loading failed! Check your data path.")
        return

    # Training configuration
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\nTraining GRU Model...")
    for epoch in range(10):  # 10 epochs
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Print training progress
        if (epoch+1) % 2 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f'Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}')

    # Save trained model weights
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/gru_weights.pth')
    print("\nTraining complete! Saved weights to models/gru_weights.pth")
    
    # Evaluation
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x.to(device))
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(batch_y.numpy().flatten())
    
    # Calculate final metrics
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    print(f"Final RMSE: {rmse:.4f}")

    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(actuals, predictions, alpha=0.3)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    
    plt.subplot(1, 2, 2)
    errors = np.array(predictions) - np.array(actuals)
    plt.hist(errors, bins=30)
    plt.xlabel('Prediction Errors')
    
    plt.suptitle('GRU Model Evaluation')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_and_save_gru()
