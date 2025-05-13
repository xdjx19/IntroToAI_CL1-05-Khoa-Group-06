import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_preprocessing import TrafficDataset

class TrafficPredictionModels:
    def __init__(self, train_loader, test_loader):
        """
        Initialize the traffic prediction models with data loaders from data_preprocessing.py
        
        :param train_loader: DataLoader for training data
        :param test_loader: DataLoader for testing data
        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # Models
        self.lstm_model = None
        self.gru_model = None
        self.random_forest_model = None
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def build_lstm_model(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
        """
        Build and compile LSTM model
        
        :param input_size: Number of input features
        :param hidden_size: Number of LSTM units
        :param num_layers: Number of LSTM layers
        :param dropout: Dropout rate
        :return: Compiled LSTM model
        """
        self.lstm_model = nn.Sequential(
            nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                    num_layers=num_layers, batch_first=True, dropout=dropout),
            nn.Linear(hidden_size, 1)
        ).to(self.device)
        
        self.lstm_model.compile(optimizer='adam', loss='mse')
        return self.lstm_model
    
    def build_gru_model(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
        """
        Build and compile GRU model
        
        :param input_size: Number of input features
        :param hidden_size: Number of GRU units
        :param num_layers: Number of GRU layers
        :param dropout: Dropout rate
        :return: Compiled GRU model
        """
        self.gru_model = nn.Sequential(
            nn.GRU(input_size=input_size, hidden_size=hidden_size, 
                   num_layers=num_layers, batch_first=True, dropout=dropout),
            nn.Linear(hidden_size, 1)
        ).to(self.device)
        
        self.gru_model.compile(optimizer='adam', loss='mse')
        return self.gru_model
    
    def train_models(self, epochs=50):
        """
        Train LSTM and GRU models
        
        :param epochs: Number of training epochs
        """
        criterion = nn.MSELoss()
        
        # Train LSTM
        if self.lstm_model is None:
            self.build_lstm_model()
        
        optimizer = torch.optim.Adam(self.lstm_model.parameters())
        
        for epoch in range(epochs):
            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs, _ = self.lstm_model[0](batch_x)
                loss = criterion(self.lstm_model[1](outputs[:, -1, :]), batch_y)
                loss.backward()
                optimizer.step()
            print(f"LSTM Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
        # Train GRU
        if self.gru_model is None:
            self.build_gru_model()
        
        optimizer = torch.optim.Adam(self.gru_model.parameters())
        
        for epoch in range(epochs):
            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs, _ = self.gru_model[0](batch_x)
                loss = criterion(self.gru_model[1](outputs[:, -1, :]), batch_y)
                loss.backward()
                optimizer.step()
            print(f"GRU Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    def evaluate_models(self):
        """
        Evaluate the performance of trained models
        
        :return: Dictionary of model performance metrics
        """
        results = {}
        criterion = nn.MSELoss()
        
        # LSTM Evaluation
        self.lstm_model.eval()
        lstm_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs, _ = self.lstm_model[0](batch_x)
                loss = criterion(self.lstm_model[1](outputs[:, -1, :]), batch_y)
                lstm_loss += loss.item()
        results['lstm'] = {'MSE': lstm_loss / len(self.test_loader)}
        
        # GRU Evaluation
        self.gru_model.eval()
        gru_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs, _ = self.gru_model[0](batch_x)
                loss = criterion(self.gru_model[1](outputs[:, -1, :]), batch_y)
                gru_loss += loss.item()
        results['gru'] = {'MSE': gru_loss / len(self.test_loader)}
        
        return results

# Example usage
if __name__ == '__main__':
    from data_preprocessing import get_data_loaders
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders("Scats_Data_October_2006.xlsx")
    
    # Initialize the traffic prediction model
    predictor = TrafficPredictionModels(train_loader, test_loader)
    
    # Train models
    predictor.train_models(epochs=5)
    
    # Evaluate models
    model_performance = predictor.evaluate_models()
    print("Model Performance:", model_performance)