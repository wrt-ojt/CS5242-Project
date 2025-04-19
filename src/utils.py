# src/utils.py
import torch
import random
import numpy as np
import os
import json
import logging

def set_seed(seed_value):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        # Might impact performance, uncomment if needed for full determinism
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def save_checkpoint(model, optimizer, epoch, filepath):
    """Saves model and optimizer state."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, filepath)
    # print(f"Checkpoint saved to {filepath}")

def load_checkpoint(model, optimizer, filepath, device):
    """Loads model and optimizer state."""
    if not os.path.exists(filepath):
        print(f"Warning: Checkpoint file not found at {filepath}")
        return None, None, -1 # Return defaults indicating failure
    checkpoint = torch.load(filepath, map_location=device)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer: # Optimizer might be None if only loading for inference
             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from {filepath}, epoch {epoch}")
        return model, optimizer, epoch
    except KeyError as e:
         print(f"Error: Checkpoint file {filepath} is missing key: {e}")
         # Try loading only model state if possible (e.g., older checkpoint)
         try:
             model.load_state_dict(torch.load(filepath, map_location=device))
             print("Loaded only model state_dict (might be older format or inference-only save).")
             return model, None, -1 # Indicate optimizer/epoch not loaded
         except Exception as load_err:
              print(f"Could not load state_dict either: {load_err}")
              return None, None, -1
    except Exception as e:
        print(f"Error loading checkpoint from {filepath}: {e}")
        return None, None, -1

def save_config(config, filepath):
    """Saves the configuration dictionary to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        # Convert non-serializable items if necessary (e.g., device object)
        serializable_config = config.copy()
        if isinstance(serializable_config.get('device'), torch.device):
             serializable_config['device'] = str(serializable_config['device'])
        json.dump(serializable_config, f, indent=4)
    print(f"Configuration saved to {filepath}")

def setup_logging(log_file):
    """Sets up logging to file and console."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() # Also print to console
        ]
    )
    logging.info("Logging setup complete.")

def save_results(results, filepath):
     """Saves evaluation results to a JSON file."""
     os.makedirs(os.path.dirname(filepath), exist_ok=True)
     with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
     logging.info(f"Results saved to {filepath}")
