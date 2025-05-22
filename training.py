import os
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import numpy as np
from data_preprocessing import get_data_loaders

# Enhanced and smaller LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, dropout=0.2, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=bidirectional)
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_factor, 1)
        self._init_weights()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # take last time step
        return out

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

# Train function
def train_model(model, train_loader, test_loader, epochs=5, lr=0.001, device='cpu'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Evaluation
        model.eval()
        test_preds, test_actuals = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                test_preds.extend(outputs.cpu().numpy())
                test_actuals.extend(batch_y.cpu().numpy())

        train_loss = np.mean(train_losses)
        test_rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Test RMSE: {test_rmse:.4f}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "Scats_Data_October_2006.xlsx")

    if not os.path.exists(data_path):
        print("Error: Data file not found.")
    else:
        train_loader, test_loader = get_data_loaders(data_path, batch_size=64, seq_len=24)

        if train_loader is not None:
            model = LSTMModel()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Training on: {device}")
            train_model(model, train_loader, test_loader, epochs=5, lr=0.001, device=device)
