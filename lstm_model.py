import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

class LSTMTrafficModel(nn.Module):
    """Enhanced LSTM model with weight saving capability"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_and_save_model():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = LSTMTrafficModel().to(device)
    
    # Load data (make sure data_preprocessing.py is in same directory)
    from data_preprocessing import get_data_loaders
    train_loader, test_loader = get_data_loaders("Scats_Data_October_2006.xlsx")
    
    if train_loader is None:
        print("Data loading failed! Check your data path.")
        return

    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\nTraining LSTM Model...")
    for epoch in range(10):  # 10 epochs
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Print progress
        if (epoch+1) % 2 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

    # Save trained weights
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/lstm_weights.pth')
    print("\nTraining complete! Saved weights to models/lstm_weights.pth")
    
    # Evaluation
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x.to(device))
            preds.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.numpy())
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    print(f"\nFinal RMSE: {rmse:.4f}")

if __name__ == "__main__":
    train_and_save_model()
