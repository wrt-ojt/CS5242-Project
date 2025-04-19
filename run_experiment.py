# run_experiment.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import logging
import json

# Ensure src is importable
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.config import CONFIG  # Import default config
from src import preprocess
from src import dataset
from src import model as model_module # Avoid conflict with variable name 'model'
from src import train
from src import evaluate
from src import utils

def override_config(config, args):
    """Overrides default config with command-line arguments."""
    for key, value in vars(args).items():
        if value is not None and key in config:
            # Special handling for list arguments if needed
            if isinstance(config[key], list) and isinstance(value, str):
                 # Example: Assuming comma-separated input for lists like classifier_hidden_layers
                 try:
                     config[key] = [int(x.strip()) for x in value.split(',')]
                 except ValueError:
                     print(f"Warning: Could not parse list argument for {key}: {value}")
            else:
                # Handle potential type mismatches (e.g., str to bool/int/float)
                try:
                    # Attempt type conversion based on default config type
                    target_type = type(config[key])
                    if target_type == bool:
                        config[key] = value.lower() in ('true', '1', 't', 'yes', 'y')
                    elif target_type == int:
                         config[key] = int(value)
                    elif target_type == float:
                         config[key] = float(value)
                    else: # Primarily string or keep as is
                        config[key] = value
                except (ValueError, TypeError) as e:
                     print(f"Warning: Could not convert argument for {key}={value} to type {target_type}. Using string. Error: {e}")
                     config[key] = value # Fallback to string or original value
    # Update dependent config values
    config["experiment_output_dir"] = os.path.join(config["output_dir"], config["experiment_name"])
    return config

def main():
    parser = argparse.ArgumentParser(description="Run Multimodal Classification Experiment")
    # Add arguments for keys in CONFIG you want to make overridable
    parser.add_argument('--experiment_name', type=str, help="Name for the experiment output folder.")
    parser.add_argument('--batch_size', type=int, help="Batch size for training and evaluation.")
    parser.add_argument('--num_epochs', type=int, help="Number of training epochs.")
    parser.add_argument('--learning_rate_head', type=float, help="Learning rate for the classifier head.")
    parser.add_argument('--learning_rate_clip', type=float, help="Learning rate for fine-tuning CLIP.")
    parser.add_argument('--weight_decay_head', type=float, help="Weight decay for the classifier head.")
    parser.add_argument('--weight_decay_clip', type=float, help="Weight decay for fine-tuning CLIP.")
    parser.add_argument('--freeze_clip', type=str, help="Freeze CLIP weights ('true' or 'false').") # Input as string for easier bool conversion
    parser.add_argument('--modality', type=str, choices=['multimodal', 'image', 'text'], help="Modality to use.")
    parser.add_argument('--use_cross_attention', type=str, help="Use cross-attention ('true' or 'false').")
    parser.add_argument('--dropout_mlp', type=float, help="Dropout rate for the MLP head.")
    parser.add_argument('--classifier_hidden_layers', type=str, help="Comma-separated list of hidden layer sizes (e.g., '1024,512'). Use '' for direct.")
    parser.add_argument('--force_preprocess', action='store_true', help="Force data preprocessing.")
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help="Device to use ('cuda' or 'cpu').")
    parser.add_argument('--num_workers', type=int, help="Number of workers for DataLoader.")
    # Add more arguments as needed

    args = parser.parse_args()
    config = override_config(CONFIG.copy(), args) # Use a copy to avoid modifying the original CONFIG dict

    # --- Setup Output and Logging ---
    exp_dir = config['experiment_output_dir']
    os.makedirs(exp_dir, exist_ok=True)
    log_file = os.path.join(exp_dir, 'logs.log')
    utils.setup_logging(log_file)
    logging.info("Starting experiment: %s", config['experiment_name'])
    logging.info("Configuration:\n%s", json.dumps(config, indent=4))
    utils.save_config(config, os.path.join(exp_dir, 'config.json'))

    # --- Set Seed ---
    utils.set_seed(config['seed'])
    logging.info(f"Random seed set to {config['seed']}")

    # --- 1. Preprocessing ---
    if config.get('force_preprocess', False): # Use .get for safety
        logging.info("Force preprocessing enabled.")
    preprocess.main(config)

    # --- 2. Create DataLoaders ---
    logging.info("Creating DataLoaders...")
    try:
        train_loader, val_loader, test_loader = dataset.create_dataloaders(config)
    except FileNotFoundError as e:
        logging.error(f"Failed to create DataLoaders: {e}. Ensure preprocessing was successful.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred creating DataLoaders: {e}")
        sys.exit(1)

    # --- 3. Initialize Model ---
    logging.info("Initializing model...")
    model = model_module.MultimodalClassifier(config)
    logging.info(f"Model loaded on device: {config['device']}")
    # print(model) # Optional: Print model structure

    # --- 4. Initialize Loss & Optimizer ---
    criterion = nn.CrossEntropyLoss()
    logging.info("Setting up optimizer...")

    # Separate parameters for different LR and Weight Decay
    head_params_ids = set()
    if hasattr(model, 'image_projection'): head_params_ids.update(map(id, model.image_projection.parameters()))
    if hasattr(model, 'text_projection'): head_params_ids.update(map(id, model.text_projection.parameters()))
    if config['use_cross_attention'] and config['modality'] == 'multimodal':
         if hasattr(model, 'img_to_txt_attention'): head_params_ids.update(map(id, model.img_to_txt_attention.parameters()))
         if hasattr(model, 'txt_to_img_attention'): head_params_ids.update(map(id, model.txt_to_img_attention.parameters()))
    if config['use_cnn_layer'] and config['modality'] == 'multimodal':
         if hasattr(model, 'conv1d'): head_params_ids.update(map(id, model.conv1d.parameters()))
         if hasattr(model, 'relu_cnn'): head_params_ids.update(map(id, model.relu_cnn.parameters())) # Assuming ReLU has no params, but safe check
    head_params_ids.update(map(id, model.classifier_head.parameters()))

    clip_params = [p for p in model.clip_model.parameters() if p.requires_grad] # Only optimize unfrozen clip params
    head_params = [p for p in model.parameters() if id(p) in head_params_ids and p.requires_grad]

    optimizer_grouped_parameters = [
        {'params': clip_params, 'lr': config['learning_rate_clip'], 'weight_decay': config['weight_decay_clip']},
        {'params': head_params, 'lr': config['learning_rate_head'], 'weight_decay': config['weight_decay_head']}
    ]

    # Filter out empty parameter groups
    optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g['params']]

    if not optimizer_grouped_parameters:
         logging.error("No parameters to optimize. Check model structure and freeze_clip setting.")
         sys.exit(1)
    else:
         logging.info(f"Optimizing {len(clip_params)} CLIP parameters with LR={config['learning_rate_clip']}, WD={config['weight_decay_clip']}")
         logging.info(f"Optimizing {len(head_params)} Head parameters with LR={config['learning_rate_head']}, WD={config['weight_decay_head']}")

    optimizer = optim.AdamW(optimizer_grouped_parameters)

    # --- 5. Training ---
    best_model_path = train.train_model(config, model, train_loader, val_loader, optimizer, criterion)

    # --- 6. Testing ---
    if best_model_path and os.path.exists(best_model_path):
        evaluate.test_model(config, model, test_loader, criterion)
    else:
        logging.warning("Best model path not found or training failed. Skipping final testing.")

    logging.info("--- Experiment Finished: %s ---", config['experiment_name'])

if __name__ == "__main__":
    main()
