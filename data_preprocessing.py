import os
import warnings
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore", message="Cannot parse header or footer*")

# Traffic dataset class
class TrafficDataset(Dataset):
    def __init__(self, sequences, targets):
        self.X = sequences
        self.y = targets

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]])

# LSTM model definition
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers,
                                  batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Use the last time step
        return self.fc(out)

# Preprocesses the data
def preprocess_data(file_path, sequence_length=24):
    try:
        df = pd.read_excel(file_path, sheet_name=1, header=1)
        df.columns = df.columns.map(lambda x: str(x).strip())

        excluded = {'SCATS Number', 'Location', 'CD_MELWAY', 'NB_LATITUDE', 'NB_LONGITUDE',
                    'HF VicRoads Internal', 'VR Internal Stat', 'VR Internal Loc',
                    'NB_TYPE_SURVEY', 'Date'}
        print("\nColumns found (excluding internal/system columns):")
        print([col for col in df.columns if col not in excluded])

        site_col = next((col for col in df.columns if 'SCATS' in col or 'CD_MELWAY' in col), None)
        location_col = next((col for col in df.columns if 'Location' in col), None)
        date_col = next((col for col in df.columns if 'Date' in col), None)

        if not all([location_col, date_col]):
            raise ValueError("Required columns like 'Date' or 'Location' not found.")

        time_cols = [col for col in df.columns if col.startswith('V') and col[1:].isdigit()]
        df = df[[location_col, date_col] + time_cols].rename(columns={
            location_col: 'Location',
            date_col: 'Date',
        })

        df['SCATS Number'] = df['Location']
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)

        df_long = df.melt(id_vars=['SCATS Number', 'Location', 'Date'],
                          value_vars=time_cols,
                          var_name='Time',
                          value_name='VehicleCount')

        df_long['Minutes'] = df_long['Time'].apply(lambda x: int(x[1:]) * 15 if x.startswith('V') else np.nan)
        df_long.dropna(subset=['Minutes'], inplace=True)
        df_long['DateTime'] = df_long['Date'] + pd.to_timedelta(df_long['Minutes'], unit='m')
        df_long['VehicleCount'] = pd.to_numeric(df_long['VehicleCount'], errors='coerce')
        df_long.dropna(subset=['VehicleCount'], inplace=True)

        # Normalize vehicle count
        scaler = MinMaxScaler()
        df_long['VehicleCount'] = scaler.fit_transform(df_long[['VehicleCount']])

        output_excel_path = os.path.join(os.path.dirname(file_path), "Updated_PrePro_Data.xlsx")
        df_long.to_excel(output_excel_path, index=False)
        print(f"\nPreprocessed long-format data saved to: {output_excel_path}")

        sequences, targets = [], []
        for scats_id in df_long['SCATS Number'].unique():
            site_data = df_long[df_long['SCATS Number'] == scats_id].sort_values('DateTime')
            values = site_data['VehicleCount'].values
            for i in range(len(values) - sequence_length):
                sequences.append(values[i:i + sequence_length])
                targets.append(values[i + sequence_length])

        X = np.array(sequences).reshape(-1, sequence_length, 1)
        y = np.array(targets)

        return X, y, scaler

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None, None, None

# Splits the data and returns data loaders
def get_data_loaders(file_path, batch_size=32, seq_len=24):
    X, y, scaler = preprocess_data(file_path, sequence_length=seq_len)
    if X is None or y is None:
        return None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = TrafficDataset(X_train, y_train)
    test_dataset = TrafficDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, scaler

# Evaluation using RMSE and MAE
def evaluate(model, test_loader, scaler):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x).squeeze()
            preds.extend(outputs.numpy())
            targets.extend(batch_y.squeeze().numpy())

    # Inverse transform to original scale
    preds_rescaled = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    targets_rescaled = scaler.inverse_transform(np.array(targets).reshape(-1, 1)).flatten()

    rmse = mean_squared_error(targets_rescaled, preds_rescaled, squared=False)
    mae = mean_absolute_error(targets_rescaled, preds_rescaled)

    print(f"\nTest RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    # --- Plot predictions ---
    plt.figure(figsize=(12, 6))
    plt.plot(targets_rescaled[:200], label='Actual', linewidth=2)
    plt.plot(preds_rescaled[:200], label='Predicted', linestyle='--', linewidth=2)
    plt.title("Vehicle Count Prediction vs Actual (First 200 test samples)")
    plt.xlabel("Time Step")
    plt.ylabel("Vehicle Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main training routine
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "Scats_Data_October_2006.xlsx")

    print(f"\nLooking for data file at: {data_path}")
    print(f"File exists: {os.path.exists(data_path)}")

    if not os.path.exists(data_path):
        print("Error: Data file not found. Please check the path.")
    else:
        train_loader, test_loader, scaler = get_data_loaders(data_path)

        if train_loader is not None:
            model = LSTMModel()
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

            for epoch in range(30):
                model.train()
                epoch_loss = 0
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    predictions = model(batch_x)
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                scheduler.step()
                print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.4f}")

            evaluate(model, test_loader, scaler)


#long short term memory (lstm model used) specifically recurrent neural network (rnn)

#scats = site id
#location = where the traffic data was recorded
#date = when the traffic data was recorded
#v00 - v95 = the traffic count from every 15 minutes for 24 hours
#ignore warning message. only occuring because of had weird formatting like a footer, merged cells, etc.

#model is trained on the traffic data, running through the dataset 5 times in epochs (check output)
#after running through each round, tells how far model preductions are from the actual traffic count

#make sure 'pip install pandas numpy torch scikit-learn openpyxl' is put in the terminal to make the inputs functional

#create a new training file for loading the data and training purposes
#improve model
#graph it as well
#early stopping
#call training py
#graph for training and testing

# Main.py

# import os
# import subprocess
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from Training import LSTMModel

# def load_scaler(mins, scales):
#     from sklearn.preprocessing import MinMaxScaler
#     scaler = MinMaxScaler()
#     scaler.min_, scaler.scale_ = 0, scales
#     scaler.data_min_ = mins
#     scaler.data_max_ = mins + scales
#     return scaler

# def evaluate_and_plot(model, X_test, y_test, scaler, train_losses):
#     model.eval()
#     with torch.no_grad():
#         predictions = model(torch.FloatTensor(X_test)).squeeze().numpy()

#     y_pred_inv = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
#     y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

#     # Test plot
#     plt.figure(figsize=(12, 6))
#     plt.plot(y_test_inv[:200], label='Actual', linewidth=2)
#     plt.plot(y_pred_inv[:200], label='Predicted', linestyle='--', linewidth=2)
#     plt.title("Test Data: Actual vs Predicted")
#     plt.xlabel("Sample Index")
#     plt.ylabel("Vehicle Count")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     # Training loss plot
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_losses, marker='o')
#     plt.title("Training Loss Over Epochs")
#     plt.xlabel("Epoch")
#     plt.ylabel("MSE Loss")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     print("Training model...")
#     subprocess.run(["python", "Training.py"], check=True)

#     print("\nEvaluating model...")
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     output_dir = os.path.join(script_dir, "model_output")

#     model = LSTMModel()
#     model.load_state_dict(torch.load(os.path.join(output_dir, "model.pt")))

#     X_test = np.load(os.path.join(output_dir, "X_test.npy"))
#     y_test = np.load(os.path.join(output_dir, "y_test.npy"))
#     train_losses = np.load(os.path.join(output_dir, "train_losses.npy"))

#     scaler_min = np.load(os.path.join(output_dir, "scaler_minmax.npy"))
#     scaler_scale = np.load(os.path.join(output_dir, "scaler_scale.npy"))
#     scaler = load_scaler(scaler_min, scaler_scale)

#     evaluate_and_plot(model, X_test, y_test, scaler, train_losses)
