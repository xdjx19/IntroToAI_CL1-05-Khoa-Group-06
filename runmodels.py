import os
import subprocess
from pathlib import Path

# Constants
PREPROCESSED_PATH = "Updated_PrePro_Data.xlsx"
PREPROCESSED_SCRIPT_PATH = "preprocessed_data.npy"
RAW_DATA_PATH = "Scats_Data_October_2006.xlsx"
MODEL_DIR = "models"
OUTPUT_DIR = "Assignments_2A"
MODEL_CHOICES = {
    '1': 'gru_model.py',
    '2': 'lstm_model.py', 
    '3': 'sae_model.py'
}

def ensure_directories_exist():
    """Create required directories if they don't exist."""
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def should_rerun_preprocessing():
    """Determine if preprocessing should be rerun based on user input."""
    if not os.path.exists(PREPROCESSED_PATH):
        print("No preprocessed data found. Running preprocessing...")
        return True

    choice = input("Preprocessed data found. Re-run preprocessing? (yes/no): ").lower()
    return choice == 'yes'

def get_model_choice():
    """Prompt user to select which model to run."""
    print("\nAvailable models:")
    for key, value in MODEL_CHOICES.items():
        print(f"{key}. {value.split('.')[0].upper()}")

    while True:
        choice = input("\nSelect model to run (1-3): ")
        if choice in MODEL_CHOICES:
            return MODEL_CHOICES[choice]
        print("Invalid choice. Please enter 1, 2, or 3.")

def run_script(script_name):
    """Execute the specified Python script."""
    subprocess.run(["python", script_name])

def main():
    # Configuration and setup
    ensure_directories_exist()

    # Data preprocessing
    if should_rerun_preprocessing():
        run_script("data_preprocessing.py")

    # Model selection and execution
    selected_model = get_model_choice()
    print(f"\nRunning {selected_model}...")
    run_script(selected_model)

if __name__ == "__main__":
    main()
