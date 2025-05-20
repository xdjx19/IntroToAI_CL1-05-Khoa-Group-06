import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Tuple, Dict, Union

# ========== Unified Evaluation Functions ==========

def evaluate_model(model: nn.Module, 
                  test_loader: torch.utils.data.DataLoader, 
                  criterion: torch.nn.Module = None,
                  device: str = 'cpu',
                  model_type: str = 'gru') -> Dict[str, float]:
    """
    Unified evaluation function for all model types (SAE, GRU, LSTM)
    
    Args:
        model: The trained model to evaluate
        test_loader: DataLoader containing test data
        criterion: Loss function (optional)
        device: Device to run evaluation on
        model_type: Type of model ('sae', 'gru', or 'lstm')
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    all_preds = []
    all_actuals = []
    total_loss = 0
    is_autoencoder = model_type.lower() == 'sae'
    
    # Default to MSE if no criterion provided
    if criterion is None:
        criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch in test_loader:
            # Handle different data formats for different models
            if model_type.lower() == 'sae':
                # SAE expects just input data (no labels)
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch_x = batch.to(device)
                batch_y = batch_x  # For autoencoder, target is input
            else:
                # GRU/LSTM expect (input, target) pairs
                batch_x, batch_y = batch[0].to(device), batch[1].to(device)
            
            # Forward pass
            outputs = model(batch_x)
            
            # Calculate loss
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            # Store predictions and actuals
            if model_type.lower() == 'sae':
                # For SAE, we compare reconstructions to original inputs
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_actuals.extend(batch_y.cpu().numpy().flatten())
            else:
                # For GRU/LSTM, we compare predictions to targets
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_actuals.extend(batch_y.cpu().numpy().flatten())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    
    # Calculate metrics
    metrics = {
        'loss': total_loss / len(test_loader),
        'mse': mean_squared_error(all_actuals, all_preds),
        'mae': mean_absolute_error(all_actuals, all_preds),
        'rmse': np.sqrt(mean_squared_error(all_actuals, all_preds)),
        'r2': r2_score(all_actuals, all_preds)
    }
    
    # Additional SAE-specific metrics
    if model_type.lower() == 'sae':
        metrics['reconstruction_error'] = np.mean(np.abs(all_actuals - all_preds))
    
    return metrics, all_preds, all_actuals

def print_evaluation_metrics(metrics: Dict[str, float], model_name: str):
    """Print formatted evaluation metrics"""
    print(f"\n{model_name.upper()} Evaluation Metrics:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric.upper():<20}: {value:.4f}")
    print("-" * 40)

def plot_model_results(train_losses: list, 
                      test_losses: list = None,
                      predictions: np.ndarray = None, 
                      actuals: np.ndarray = None,
                      model_name: str = "Model"):
    """
    Unified plotting function for all models
    
    Args:
        train_losses: List of training losses per epoch
        test_losses: Optional list of test losses per epoch
        predictions: Model predictions
        actuals: Ground truth values
        model_name: Name of model for title
    """
    plt.figure(figsize=(15, 5))
    
    # Plot training/test losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    if test_losses is not None:
        plt.plot(test_losses, label='Test Loss')
    plt.title(f'{model_name} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot predictions vs actuals if provided
    if predictions is not None and actuals is not None:
        plt.subplot(1, 2, 2)
        plt.scatter(actuals, predictions, alpha=0.3)
        plt.plot([actuals.min(), actuals.max()], 
                 [actuals.min(), actuals.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name} Predictions vs Actuals')
    
    plt.tight_layout()
    plt.show()

# ========== Model-Specific Wrappers ==========

def evaluate_sae(model: nn.Module, 
                test_loader: torch.utils.data.DataLoader, 
                device: str = 'cpu') -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Wrapper for SAE evaluation"""
    metrics, preds, actuals = evaluate_model(model, test_loader, 
                                           device=device, model_type='sae')
    print_evaluation_metrics(metrics, "Sparse Autoencoder")
    return metrics, preds, actuals

def evaluate_gru(model: nn.Module, 
                test_loader: torch.utils.data.DataLoader, 
                criterion: torch.nn.Module = None,
                device: str = 'cpu') -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Wrapper for GRU evaluation"""
    metrics, preds, actuals = evaluate_model(model, test_loader, criterion,
                                           device=device, model_type='gru')
    print_evaluation_metrics(metrics, "GRU Model")
    return metrics, preds, actuals

def evaluate_lstm(model: nn.Module, 
                 test_loader: torch.utils.data.DataLoader, 
                 criterion: torch.nn.Module = None,
                 device: str = 'cpu') -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Wrapper for LSTM evaluation"""
    metrics, preds, actuals = evaluate_model(model, test_loader, criterion,
                                           device=device, model_type='lstm')
    print_evaluation_metrics(metrics, "LSTM Model")
    return metrics, preds, actuals

# ========== Example Usage ==========
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader = get_data_loaders("Scats_Data_October_2006.xlsx")
    
    if train_loader and test_loader:
        # Initialize models directly (no import needed)
        sae = SparseAutoencoder(input_dim=64).to(device)
        gru = GRUTrafficModel().to(device)
        lstm = LSTMTrafficModel().to(device)
        
        # Evaluate SAE
        print("\nEvaluating SAE...")
        sae_metrics, sae_preds, sae_actuals = evaluate_sae(sae, test_loader, device)
        plot_model_results([], predictions=sae_preds, actuals=sae_actuals, 
                          model_name="SAE Reconstructions")
        
        # Evaluate GRU
        print("\nEvaluating GRU...")
        gru_metrics, gru_preds, gru_actuals = evaluate_gru(gru, test_loader, 
                                                          nn.MSELoss(), device)
        plot_model_results([], predictions=gru_preds, actuals=gru_actuals,
                          model_name="GRU Predictions")
        
        # Evaluate LSTM
        print("\nEvaluating LSTM...")
        lstm_metrics, lstm_preds, lstm_actuals = evaluate_lstm(lstm, test_loader,
                                                             nn.MSELoss(), device)
        plot_model_results([], predictions=lstm_preds, actuals=lstm_actuals,
                          model_name="LSTM Predictions")





# import os
# import sys
# import importlib.util
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split

# class TrafficEvaluator:
#     def __init__(self, data_path="Scats_Data_October_2006.xlsx"):
#         self.data_path = data_path
#         self.sequence_length = 24
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.models = {}
#         self.results = {}

#     def load_data(self):
#         """Load and preprocess traffic data"""
#         try:
#             df = pd.read_excel(self.data_path, sheet_name=1, header=1)
#             df.columns = df.columns.str.strip()
            
#             time_cols = [col for col in df.columns if col.startswith('V') and col[1:].isdigit()]
#             location_col = next(col for col in df.columns if 'Location' in col)
#             date_col = next(col for col in df.columns if 'Date' in col)
#             scats_col = next((col for col in df.columns if 'SCATS' in col or 'CD_MELWAY' in col), 'Location')
            
#             df = df[[scats_col, location_col, date_col] + time_cols]
#             df = df.rename(columns={scats_col: 'SCATS', location_col: 'Location', date_col: 'Date'})
            
#             df_long = df.melt(
#                 id_vars=['SCATS', 'Location', 'Date'],
#                 value_vars=time_cols,
#                 var_name='Time',
#                 value_name='VehicleCount'
#             )
            
#             df_long['Minutes'] = df_long['Time'].str[1:].astype(int) * 15
#             df_long['DateTime'] = pd.to_datetime(df_long['Date']) + pd.to_timedelta(df_long['Minutes'], unit='m')
#             df_long['VehicleCount'] = pd.to_numeric(df_long['VehicleCount'], errors='coerce')
            
#             return df_long.dropna(subset=['VehicleCount', 'DateTime'])
            
#         except Exception as e:
#             print(f"Data loading error: {str(e)}")
#             return None

#     def create_sequences(self, clean_df):
#         sequences, targets = [], []
#         for scats_id in clean_df['SCATS'].unique():
#             site_data = clean_df[clean_df['SCATS'] == scats_id].sort_values('DateTime')
#             values = site_data['VehicleCount'].values
#             if len(values) < self.sequence_length + 1:
#                 continue  # Skip if not enough data
#             for i in range(len(values) - self.sequence_length):
#                 sequences.append(values[i:i + self.sequence_length])
#                 targets.append(values[i + self.sequence_length])
#         return np.array(sequences), np.array(targets)

#     def load_models(self):
#         """Load models from the same directory as this script, weights from models directory"""
#         model_files = {
#             'LSTM': 'lstm_model.py',
#             'GRU': 'gru_model.py', 
#             'SAE': 'sae_model.py'
#         }
        
#         script_dir = os.path.dirname(os.path.abspath(__file__))
#         models_dir = os.path.join(script_dir, 'models')
        
#         for name, file in model_files.items():
#             try:
#                 # Load model definition from current directory
#                 filepath = os.path.join(script_dir, file)
                
#                 if not os.path.exists(filepath):
#                     print(f"Model file not found: {filepath}")
#                     self.models[name] = None
#                     continue
                    
#                 module_name = file.replace('.py', '')
#                 spec = importlib.util.spec_from_file_location(module_name, filepath)
#                 module = importlib.util.module_from_spec(spec)
#                 spec.loader.exec_module(module)
                
#                 # Get model class
#                 model_class = getattr(module, 
#                                     'LSTMTrafficModel' if name == 'LSTM' else
#                                     'GRUTrafficModel' if name == 'GRU' else 
#                                     'SparseAutoencoder')
                
#                 # Initialize model
#                 model = model_class()
                
#                 # Load weights from models directory
#                 weights_file = f"{name.lower()}_weights.pth"
#                 weights_path = os.path.join(models_dir, weights_file)
#                 if os.path.exists(weights_path):
#                     model.load_state_dict(torch.load(weights_path))
#                 else:
#                     print(f"Weights file not found: {weights_path}")
#                     self.models[name] = None
#                     continue
                
#                 model.to(self.device)
#                 model.eval()
#                 self.models[name] = model
                
#             except Exception as e:
#                 print(f"Error loading {name}: {str(e)}")
#                 self.models[name] = None

#     def evaluate(self):
#         clean_data = self.load_data()
#         if clean_data is None:
#             return False
            
#         X, y = self.create_sequences(clean_data)
#         if len(X) == 0:
#             print("Error: No valid sequences created")
#             return False
            
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
#         print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")  # Debug shape
        
#         X_test = torch.FloatTensor(X_test).unsqueeze(-1).to(self.device)
#         y_test = torch.FloatTensor(y_test).to(self.device)
        
#         for name, model in self.models.items():
#             if model is None:
#                 continue
                
#             with torch.no_grad():
#                 outputs = model(X_test)
#                 print(f"Model {name} output shape: {outputs.shape}")  # Debug output shape
                
#                 # Handle different model output types
#                 if outputs.dim() == 3:  # Sequence output
#                     preds = outputs[:, -1, :].cpu().numpy().flatten()  # Take last timestep
#                 else:
#                     preds = outputs.cpu().numpy().flatten()
                
#                 y_test_np = y_test.cpu().numpy().flatten()
                
#                 # Verify shapes match
#                 if len(preds) != len(y_test_np):
#                     print(f"Shape mismatch in {name}: predictions {preds.shape} vs targets {y_test_np.shape}")
#                     continue
                
#                 self.results[name] = {
#                     'MAE': mean_absolute_error(y_test_np, preds),
#                     'RMSE': np.sqrt(mean_squared_error(y_test_np, preds)),
#                     'R2': r2_score(y_test_np, preds)
#                 }
                
#                 # Visualization
#                 plt.figure(figsize=(12, 5))
#                 plt.subplot(1, 2, 1)
#                 plt.scatter(y_test_np, preds, alpha=0.5)
#                 plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--')
#                 plt.title(f'{name} Predictions')
                
#                 plt.subplot(1, 2, 2)
#                 errors = preds - y_test_np
#                 plt.hist(errors, bins=30)
#                 plt.title(f'{name} Error Distribution')
                
#                 plt.tight_layout()
#                 plt.savefig(f'{name}_evaluation.png')
#                 plt.close()
        
#         pd.DataFrame(self.results).T.to_csv('model_results.csv')
#         return True

# if __name__ == "__main__":
#     evaluator = TrafficEvaluator()
#     evaluator.load_models()
    
#     if evaluator.evaluate():
#         print("\nEvaluation Results:")
#         print(pd.DataFrame(evaluator.results).T)
#     else:
#         print("Evaluation failed")