# export_onnx.py
import torch
import torch.onnx
import os
import argparse
import json

# Make src importable
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.config import CONFIG # Import default config
from src.model import MultimodalClassifier
# Import the dummy input creation function from visualize_model.py
# Ensure visualize_model.py is in the same directory or accessible via PYTHONPATH
try:
    from visualize_model import create_dummy_input
except ImportError:
    print("Error: visualize_model.py not found or create_dummy_input function missing.")
    print("Please ensure visualize_model.py exists and contains the create_dummy_input function.")
    sys.exit(1)

# --- Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX format")
    parser.add_argument('--weights_path', type=str, required=True, help="Path to the saved model weights (.pth file, e.g., output/exp_name/best_model.pth)")
    parser.add_argument('--output_path', type=str, default="output/model.onnx", help="Path to save the exported ONNX model.")
    parser.add_argument('--config_overrides', type=str, help="JSON string to override config keys affecting model structure, e.g., '{\"modality\":\"image\"}'")
    parser.add_argument('--opset_version', type=int, default=11, help="ONNX opset version to use for export.")
    args = parser.parse_args()

    config = CONFIG.copy() # Start with default config

    # --- Apply Config Overrides (Optional) ---
    # Ensure overrides match the structure used when saving the weights
    if args.config_overrides:
        try:
            overrides = json.loads(args.config_overrides)
            config.update(overrides)
            print("Applied config overrides for model instantiation:", overrides)
        except json.JSONDecodeError:
            print("Warning: Could not parse config_overrides JSON string. Using default config.")

    device = torch.device('cpu') # ONNX export is typically done on CPU

    # --- Instantiate Model ---
    print("Initializing model architecture...")
    # Ensure model uses the potentially overridden config that matches the saved weights
    model = MultimodalClassifier(config)
    model.to(device) # Move model to CPU
    model.eval() # Set to evaluation mode

    # --- Load Weights ---
    if not os.path.exists(args.weights_path):
        print(f"Error: Weights file not found at {args.weights_path}")
        sys.exit(1)

    print(f"Loading weights from {args.weights_path}...")
    try:
        # Load state dict (adjust map_location if weights were saved on GPU)
        state_dict = torch.load(args.weights_path, map_location=device)
        # Handle potential DataParallel wrapper if weights were saved that way
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict: # Check if it's a checkpoint file
             state_dict = state_dict['model_state_dict']

        # Remove 'module.' prefix if saved using DataParallel
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)

    # --- Create Dummy Input ---
    print("Creating dummy input...")
    # Use the same config that the loaded model was trained with/defined by
    dummy_input_dict = create_dummy_input(config, device)
    # ONNX export often requires input as a tuple or list of tensors,
    # matching the order expected by the model's forward method if it were
    # to accept positional arguments instead of a dict.
    # Let's try passing the dictionary directly first, as some opsets support it.
    # If it fails, we might need to modify the model's forward or pass a tuple.
    # dummy_input_tuple = (dummy_input_dict['pixel_values'],
    #                      dummy_input_dict['input_ids'],
    #                      dummy_input_dict['attention_mask'])

    # --- Define Input/Output Names ---
    # These names will appear in the Netron graph
    input_names = list(dummy_input_dict.keys()) # e.g., ['pixel_values', 'input_ids', 'attention_mask']
    output_names = ["logits"] # Name for the final output tensor

    # --- Ensure Output Directory Exists ---
    output_dir = os.path.dirname(args.output_path)
    if output_dir: # Only create if path includes a directory
        os.makedirs(output_dir, exist_ok=True)

    # --- Export to ONNX ---
    print(f"Exporting model to ONNX format at {args.output_path} (Opset: {args.opset_version})...")
    try:
        torch.onnx.export(
            model,
            dummy_input_dict, # Pass the dictionary as input
            args.output_path,
            export_params=True,        # Store trained weights within the file
            opset_version=args.opset_version, # ONNX version
            do_constant_folding=True,  # Optimization
            input_names=input_names,   # Model input names
            output_names=output_names, # Model output names
            # Optional: Define dynamic axes if batch size or sequence length can vary
            dynamic_axes={
                'pixel_values': {0: 'batch_size'},
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size'}
            }
        )
        print("-" * 80)
        print(f"Model successfully exported to: {args.output_path}")
        print("You can now open this file using Netron.")
        print("-" * 80)
    except Exception as e:
        print(f"\n{'='*30} ONNX Export Failed {'='*30}")
        print(f"Error: {e}")
        print("\nCommon issues:")
        print("- Unsupported PyTorch operations in the specified Opset version.")
        print("- Model forward pass expecting a different input format (e.g., tuple vs. dict). Try modifying forward or input format.")
        print("- Issues with dynamic shapes or control flow within the model.")
        print(f"{'='*80}")
        sys.exit(1)

    print("ONNX export script finished.")

