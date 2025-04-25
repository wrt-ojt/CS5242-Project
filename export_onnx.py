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
# No longer importing from visualize_model.py

# --- Dummy Input Creation Function (Copied here) ---
def create_dummy_input(config, device):
    """Creates a dummy input batch matching model expectations."""
    batch_size = 2 # Use a small batch size for export
    max_len = config['max_token_length']
    # CLIP image processor typically outputs (B, C, H, W), e.g., (B, 3, 224, 224)
    dummy_pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
    # Use actual vocab size if known, otherwise a reasonable default
    vocab_size = config.get('vocab_size', 49408) # CLIP's vocab size
    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, max_len), dtype=torch.long).to(device)
    dummy_attention_mask = torch.ones(batch_size, max_len, dtype=torch.long).to(device)

    dummy_batch = {
        'pixel_values': dummy_pixel_values,
        'input_ids': dummy_input_ids,
        'attention_mask': dummy_attention_mask
    }
    return dummy_batch

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
    model = MultimodalClassifier(config)
    model.to(device)
    model.eval()

    # --- Load Weights ---
    if not os.path.exists(args.weights_path):
        print(f"Error: Weights file not found at {args.weights_path}")
        sys.exit(1)

    print(f"Loading weights from {args.weights_path}...")
    try:
        state_dict = torch.load(args.weights_path, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
             state_dict = state_dict['model_state_dict']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)

    # --- Create Dummy Input ---
    print("Creating dummy input...")
    # Now calls the function defined within this script
    dummy_input_dict = create_dummy_input(config, device)

    # --- Define Input/Output Names ---
    input_names = list(dummy_input_dict.keys())
    output_names = ["logits"]

    # --- Ensure Output Directory Exists ---
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # --- Export to ONNX ---
    print(f"Exporting model to ONNX format at {args.output_path} (Opset: {args.opset_version})...")
    try:
        torch.onnx.export(
            model,
            dummy_input_dict, # Pass the dictionary
            args.output_path,
            export_params=True,
            opset_version=args.opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
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
        print("- Model forward pass expecting a different input format (e.g., tuple vs. dict).")
        print("- Issues with dynamic shapes or control flow within the model.")
        print(f"{'='*80}")
        sys.exit(1)

    print("ONNX export script finished.")
