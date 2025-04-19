# src/config.py
import torch
import os

# --- Core Paths ---
# Assume the script runs from the root directory where label.csv, raw_data/ etc. exist
# Or adjust these paths as needed (e.g., using absolute paths or environment variables)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Get project root
RAW_DATA_DIR = os.path.join(ROOT_DIR, "raw_data")
LABEL_FILE = os.path.join(ROOT_DIR, "label.csv")
PREPROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "preprocessed_data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

CONFIG = {
    # --- Data & Paths ---
    "raw_data_dir": RAW_DATA_DIR,
    "label_file": LABEL_FILE,
    "preprocessed_data_dir": PREPROCESSED_DATA_DIR, # Base directory for preprocessed splits
    "output_dir": OUTPUT_DIR,         # Base directory for experiment outputs
    "experiment_name": "default_experiment", # Subdirectory for current run's outputs

    # --- Preprocessing ---
    "force_preprocess": False, # Set to True to always rerun preprocessing

    # --- Model ---
    "clip_model_name": "openai/clip-vit-base-patch32",
    "modality": "multimodal", # 'multimodal', 'image', 'text'
    "freeze_clip": True,
    "projection_dim": None, # Set to an int (e.g., 512) to add linear projection after CLIP features
    "use_cross_attention": True,
    "num_attention_heads": 8,
    "use_cnn_layer": False, # Conv1D might be less effective now, default off
    "cnn_out_channels_ratio": 0.5, # Ratio to determine CNN output channels if used
    "classifier_hidden_layers": [1024, 512], # List of hidden layer sizes for MLP head [] means direct
    "num_classes": 3,

    # --- Training ---
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 128, # Reduced from 256 for potentially faster CPU processing / less bottleneck
    "num_epochs": 20, # Increased epochs, relying on early stopping
    "learning_rate_clip": 1e-6,
    "learning_rate_head": 1e-4,
    "weight_decay_clip": 0.01, # Weight decay for regularization
    "weight_decay_head": 0.01,
    "dropout_attention": 0.1, # Dropout in attention layers
    "dropout_mlp": 0.4,       # Increased dropout in MLP head for regularization
    "early_stopping_patience": 5, # Stop if val loss doesn't improve for 5 epochs

    # --- Dataloader ---
    "num_workers": 4, # Adjust based on your system's CPU cores

    # --- Data Split ---
    "val_split_ratio": 0.15,
    "test_split_ratio": 0.15,
    "seed": 42,

    # --- CLIP Processor ---
    "max_token_length": 77,

    # --- Misc ---
    "label_map": {"negative": 0, "neutral": 1, "positive": 2},
}

# Calculated config values
CONFIG["inv_label_map"] = {v: k for k, v in CONFIG["label_map"].items()}
CONFIG["experiment_output_dir"] = os.path.join(CONFIG["output_dir"], CONFIG["experiment_name"])
