import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Tuple, Dict, Union
import os
import uuid
import importlib.util
import pandas as pd

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define SparseAutoencoder to match train_models.py
class SparseAutoencoder(nn.Module):
    """SAE with proper shape handling"""
    def __init__(self, input_dim=24, encoding_dim=12, hidden_dim=48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim))
        self.predict = nn.Linear(encoding_dim, 1)  # Included to match sae_weights.pth
        
    def forward(self, x):
        original_shape = x.shape
        if len(original_shape) == 3:
            x = x.view(original_shape[0], -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        if len(original_shape) == 3:
            decoded = decoded.view(original_shape)
        return decoded

def get_vehicle_count_range(file_path: str) -> Tuple[float, float]:
    """Extract min and max vehicle counts from Excel for inverse scaling"""
    try:
        df = pd.read_excel(file_path, sheet_name=1, header=1)
        df.columns = df.columns.map(lambda x: str(x).strip())
        time_cols = [col for col in df.columns if col.startswith('V') and col[1:].isdigit()]
        vehicle_counts = df[time_cols].values.flatten()
        vehicle_counts = pd.to_numeric(vehicle_counts, errors='coerce')
        vehicle_counts = vehicle_counts[~np.isnan(vehicle_counts)]
        min_val = vehicle_counts.min()
        max_val = vehicle_counts.max()
        print(f"Vehicle count range: min={min_val:.2f}, max={max_val:.2f}")
        return min_val, max_val
    except Exception as e:
        print(f"Error computing vehicle count range: {str(e)}")
        return None, None

def evaluate_model(model: nn.Module, 
                  test_loader: torch.utils.data.DataLoader, 
                  criterion: torch.nn.Module,
                  device: str,
                  model_type: str,
                  min_val: float,
                  max_val: float) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Unified evaluation function for all model types (SAE, GRU, LSTM)
    
    Args:
        model: The trained model to evaluate
        test_loader: DataLoader containing test data
        criterion: Loss function
        device: Device to run evaluation on
        model_type: Type of model ('sae', 'gru', or 'lstm')
        min_val: Minimum vehicle count for inverse scaling
        max_val: Maximum vehicle count for inverse scaling
        
    Returns:
        Tuple of (metrics dictionary, predictions, actuals) in vehicle counts
    """
    model.eval()
    all_preds = []
    all_actuals = []
    total_loss = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            if model_type.lower() == 'sae':
                batch_x_flat = batch_x.view(batch_x.size(0), -1)
                encoding = model.encoder(batch_x_flat)
                outputs = model.predict(encoding)
            else:
                outputs = model(batch_x)
            
            if outputs.shape != batch_y.shape:
                print(f"Shape mismatch in {model_type}: outputs {outputs.shape}, targets {batch_y.shape}")
                return {}, np.array([]), np.array([])
            
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            outputs_np = outputs.cpu().numpy()
            batch_y_np = batch_y.cpu().numpy()
            if min_val is not None and max_val is not None:
                outputs_np = outputs_np * (max_val - min_val) + min_val
                batch_y_np = batch_y_np * (max_val - min_val) + min_val
            
            # Clip negative predictions (traffic counts can't be negative)
            outputs_np = np.maximum(outputs_np, 0)
            batch_y_np = np.maximum(batch_y_np, 0)
            
            all_preds.extend(outputs_np)
            all_actuals.extend(batch_y_np)
    
    all_preds = np.array(all_preds).reshape(-1)
    all_actuals = np.array(all_actuals).reshape(-1)
    
    metrics = {
        'mse': mean_squared_error(all_actuals, all_preds),
        'mae': mean_absolute_error(all_actuals, all_preds),
        'rmse': np.sqrt(mean_squared_error(all_actuals, all_preds)),
        'r2': r2_score(all_actuals, all_preds)
    }
    
    print(f"{model_type.upper()} Predictions: min={all_preds.min():.2f}, max={all_preds.max():.2f}")
    print(f"{model_type.upper()} Actuals: min={all_actuals.min():.2f}, max={all_actuals.max():.2f}")
    if metrics['mse'] > 1000000 or metrics['r2'] < -10:
        print(f"Warning: {model_type.upper()} metrics may indicate issues (MSE={metrics['mse']:.2f}, R2={metrics['r2']:.2f})")
    
    return metrics, all_preds, all_actuals

def print_evaluation_metrics(metrics: Dict[str, float], model_name: str):
    """Print formatted evaluation metrics"""
    print(f"\n{model_name.upper()} Evaluation Metrics (Vehicle Counts):")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric.upper():<20}: {value:.4f}")
    print("-" * 40)

def plot_model_comparison(metrics_dict: Dict[str, Dict[str, float]], results_dir: str, artifact_id: str):
    """Plot bar chart comparing MSE, MAE, RMSE, R² across models"""
    models = list(metrics_dict.keys())
    metrics = ['mse', 'mae', 'rmse', 'r2']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [metrics_dict[model][metric] for model in models]
        if metric != 'r2':
            values = [max(v, 1e-10) for v in values]
            axes[i].bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            axes[i].set_yscale('log')
            axes[i].set_ylabel(f'{metric.upper()} (log scale)')
        else:
            axes[i].bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            axes[i].set_ylabel(metric.upper())
        axes[i].set_title(f'{metric.upper()} Comparison (Vehicle Counts)')
        axes[i].set_xlabel('Model')
        axes[i].grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(results_dir, f'model_comparison_{artifact_id}.png'), dpi=300)
    except Exception as e:
        print(f"Error saving model comparison plot: {str(e)}")
    plt.close()

def plot_predictions_vs_actual(predictions_dict: Dict[str, np.ndarray], 
                              actuals: np.ndarray, 
                              results_dir: str,
                              artifact_id: str):
    """Plot predicted vs. actual traffic volumes for each model"""
    plt.figure(figsize=(12, 6))
    sample_size = min(100, len(actuals))
    plt.plot(actuals[:sample_size], label='Actual', color='black', linestyle='--', linewidth=2)
    for model_name, preds in predictions_dict.items():
        plt.plot(preds[:sample_size], label=f'{model_name} Predicted', alpha=0.7, linewidth=1.5)
    plt.xlabel('Sample Index')
    plt.ylabel('Traffic Volume (Vehicle Counts)')
    plt.title('Predicted vs. Actual Traffic Volumes (Vehicle Counts)')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(results_dir, f'predictions_vs_actual_{artifact_id}.png'), dpi=300)
    except Exception as e:
        print(f"Error saving predictions vs actual plot: {str(e)}")
    plt.close()

def load_model_module(model_name: str, file_path: str):
    """Dynamically load a model module from a file"""
    try:
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            raise ImportError(f"Cannot load spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error loading {model_name} module from {file_path}: {str(e)}")
        return None

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create results directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'model_evaluation_results')
    try:
        os.makedirs(results_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating results directory {results_dir}: {str(e)}")
        return
    
    # Load vehicle count range for inverse scaling
    file_path = os.path.join(script_dir, "Scats_Data_October_2006.xlsx")
    min_val, max_val = get_vehicle_count_range(file_path)
    if min_val is None or max_val is None:
        print("Cannot proceed without valid vehicle count range")
        return
    
    # Load data
    try:
        import data_preprocessing
        if not hasattr(data_preprocessing, 'get_data_loaders'):
            raise AttributeError("get_data_loaders not found in data_preprocessing")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        train_loader, test_loader = data_preprocessing.get_data_loaders(
            file_path=file_path,
            batch_size=32,
            seq_len=24
        )
    except Exception as e:
        print(f"Error loading data_preprocessing.get_data_loaders: {str(e)}")
        print("Please verify data_preprocessing.py contains a working get_data_loaders function")
        print(f"Ensure Scats_Data_October_2006.xlsx is in {script_dir}")
        return
    
    if train_loader is None or test_loader is None:
        print("Error: Data loaders are empty or invalid")
        return
    
    # Verify data shapes
    try:
        for batch_x, batch_y in test_loader:
            print(f"Test loader shapes: inputs {batch_x.shape}, targets {batch_y.shape}")
            break
    except Exception as e:
        print(f"Error accessing test_loader: {str(e)}")
        return
    
    # Load model modules
    model_files = {
        'LSTM': 'lstm_model.py',
        'GRU': 'gru_model.py',
        'SAE': 'sae_model.py'
    }
    
    models = {}
    for name, file_name in model_files.items():
        if name == 'SAE':
            models[name] = SparseAutoencoder(input_dim=24, encoding_dim=12, hidden_dim=48)
            continue
        
        file_path = os.path.join(script_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Model file not found: {file_path}")
            models[name] = None
            continue
        
        module = load_model_module(name, file_path)
        if module is None:
            models[name] = None
            continue
        
        try:
            model_class = getattr(module, 
                                'LSTMTrafficModel' if name == 'LSTM' else
                                'GRUTrafficModel' if name == 'GRU' else 
                                'SparseAutoencoder')
            model = model_class(input_size=1, hidden_size=64, num_layers=2, output_size=1)
        except Exception as e:
            print(f"Error initializing {name} model: {str(e)}")
            models[name] = None
            continue
        
        models[name] = model
    
    # Load pre-trained weights
    weights_dirs = [os.path.join(script_dir, 'models')]
    for name, model in models.items():
        if model is None:
            continue
        
        weights_file = f"{name.lower()}_weights.pth"
        weights_path = None
        for dir_path in weights_dirs:
            potential_path = os.path.join(dir_path, weights_file)
            if os.path.exists(potential_path):
                weights_path = potential_path
                break
        
        if weights_path:
            try:
                state_dict = torch.load(weights_path, map_location=device)
                print(f"Loading weights for {name} from {weights_path}")
                print(f"State dict keys: {list(state_dict.keys())}")
                model_keys = list(model.state_dict().keys())
                print(f"Model expected keys: {model_keys}")
                model.load_state_dict(state_dict)
                print(f"Successfully loaded weights for {name}")
            except Exception as e:
                print(f"Error loading weights for {name}: {str(e)}")
                print(f"Expected keys: {model_keys}")
                print(f"Provided keys: {list(state_dict.keys())}")
                models[name] = None
                continue
        else:
            print(f"Warning: Weights file {weights_file} not found in {weights_dirs}")
            models[name] = None
            continue
        
        model.to(device)
        model.eval()
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Evaluate models
    metrics_dict = {}
    predictions_dict = {}
    actuals = None
    
    for name, model in models.items():
        if model is None:
            print(f"Skipping evaluation for {name} due to loading error")
            continue
        
        print(f"\nEvaluating {name}...")
        metrics, preds, acts = evaluate_model(
            model, test_loader, criterion, device, name.lower(), min_val, max_val
        )
        
        if not metrics:
            print(f"Evaluation failed for {name}")
            continue
        
        metrics_dict[name] = metrics
        predictions_dict[name] = preds
        if name == 'LSTM':
            actuals = acts
        
        print_evaluation_metrics(metrics, name)
    
    if not metrics_dict:
        print("No models were successfully evaluated")
        return
    
    # Generate visualizations
    artifact_id = str(uuid.uuid4())
    plot_model_comparison(metrics_dict, results_dir, artifact_id)
    plot_predictions_vs_actual(predictions_dict, actuals, results_dir, artifact_id)
    
    # Save results to CSV
    try:
        pd.DataFrame(metrics_dict).T.to_csv(os.path.join(results_dir, f'model_results_{artifact_id}.csv'))
        print(f"\nResults saved to {os.path.join(results_dir, f'model_results_{artifact_id}.csv')}")
        print(f"Visualizations saved to {os.path.join(results_dir, f'model_comparison_{artifact_id}.png')} and {os.path.join(results_dir, f'predictions_vs_actual_{artifact_id}.png')}")
    except Exception as e:
        print(f"Error saving CSV: {str(e)}")

if __name__ == "__main__":
    main()
