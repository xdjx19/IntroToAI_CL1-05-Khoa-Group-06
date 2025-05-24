import os
import torch
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.metrics import mean_squared_error
import numpy as np
from data_preprocessing import get_data_loaders
import matplotlib.pyplot as plt


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
        out = self.fc(out[:, -1, :])  # last time step
        return out

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)


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
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Test RMSE: {test_rmse:.4f}")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(test_actuals[:100], label='Actual', linewidth=2)
    plt.plot(test_preds[:100], label='Predicted', linewidth=2)
    plt.title("Predicted vs Actual Vehicle Count (Normalized)")
    plt.xlabel("Sample Index")
    plt.ylabel("Normalized Vehicle Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# GUI Setup
def run_gui():
    def load_file():
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            entry_path.delete(0, tk.END)
            entry_path.insert(0, file_path)

    def start_training():
        path = entry_path.get()
        if not os.path.exists(path):
            messagebox.showerror("Error", "File not found.")
            return

        train_loader, test_loader = get_data_loaders(path, batch_size=64, seq_len=24)

        if train_loader is None:
            messagebox.showerror("Error", "Data loading failed.")
            return

        model = LSTMModel()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Training on: {device}")
        train_model(model, train_loader, test_loader, epochs=5, lr=0.001, device=device)

    root = tk.Tk()
    root.title("Traffic Model Trainer")

    tk.Label(root, text="Excel File Path:").pack(padx=10, pady=(10, 0))

    entry_path = tk.Entry(root, width=50)
    default_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Scats_Data_October_2006.xlsx")
    entry_path.insert(0, default_file)
    entry_path.pack(padx=10)

    tk.Button(root, text="Browse", command=load_file).pack(pady=5)
    tk.Button(root, text="Start Training", command=start_training, bg="green", fg="white").pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    run_gui()