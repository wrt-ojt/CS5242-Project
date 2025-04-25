# export_onnx.py
import torch
import torch.onnx
import os
import argparse
import json

# Make src importable
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import default config as a fallback
from src.config import CONFIG as DEFAULT_CONFIG
from src.model import MultimodalClassifier

# --- Dummy Input Creation Function ---
def create_dummy_input(config, device):
    """Creates a dummy input batch matching model expectations."""
    batch_size = 2 # Use a small batch size for export
    max_len = config['max_token_length']
    dummy_pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
    vocab_size = config.get('vocab_size', 49408) # CLIP's vocab size
    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, max_len), dtype=torch.long).to(device)
    dummy_attention_mask = torch.ones(batch_size, max_len, dtype=torch.long).to(device)
    dummy_batch = {
        'pixel_values': dummy_pixel_values,
        'input_ids': dummy_input_ids,
        'attention_mask': dummy_attention_mask
    }
    return dummy_batch

# --- Function to load config from JSON ---
def load_config_from_experiment(weights_path):
    """Loads config.json from the same directory as the weights file."""
    config_path = os.path.join(os.path.dirname(weights_path), 'config.json')
    if not os.path.exists(config_path):
        print(f"Warning: config.json not found at {config_path}. Falling back to default config.")
        return DEFAULT_CONFIG.copy() # Return a copy of the default
    try:
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        print(f"Loaded configuration from {config_path}")
        # Ensure essential keys exist, potentially merging with defaults for safety
        # This merge is basic, adjust if complex merging is needed
        final_config = DEFAULT_CONFIG.copy()
        final_config.update(loaded_config) # Loaded values overwrite defaults
        return final_config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}. Falling back to default config.")
        return DEFAULT_CONFIG.copy()

# --- Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX format")
    parser.add_argument('--weights_path', type=str, required=True, help="Path to the saved model weights (.pth file, e.g., output/exp_name/best_model.pth)")
    parser.add_argument('--output_path', type=str, default="output/model.onnx", help="Path to save the exported ONNX model.")
    parser.add_argument('--config_overrides', type=str, help="JSON string to override keys from the loaded config.json, e.g., '{\"dropout_mlp\":0.0}'")
    parser.add_argument('--opset_version', type=int, default=11, help="ONNX opset version to use for export.")
    args = parser.parse_args()

    # --- Load Config Associated with Weights ---
    config = load_config_from_experiment(args.weights_path)

    # --- Apply Manual Overrides (Optional) ---
    # These overrides will modify the config loaded from config.json
    if args.config_overrides:
        try:
            overrides = json.loads(args.config_overrides)
            config.update(overrides)
            print("Applied manual config overrides:", overrides)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse config_overrides JSON string '{args.config_overrides}'.")

    # Ensure device is CPU for export, overriding loaded config if necessary
    config['device'] = 'cpu'
    device = torch.device(config['device'])
    print(f"Using effective configuration for export (device forced to CPU):\n{json.dumps(config, indent=2)}")


    # --- Instantiate Model ---
    print("Initializing model architecture based on loaded config...")
    # Model is now created using the config loaded from the experiment's directory
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
        # Set strict=False initially to see if only unexpected keys are the issue
        # If size mismatches persist, strict=True (default) will fail as expected
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("Load State Dict Report:")
        if missing_keys:
            print("  Missing keys in model:", missing_keys)
        if unexpected_keys:
            print("  Unexpected keys in state_dict:", unexpected_keys)
        if not missing_keys and not unexpected_keys:
             print("  Weights loaded successfully (strict=False check passed).")
        # Optionally, try strict=True again if needed for absolute certainty
        # model.load_state_dict(state_dict, strict=True)
        # print("Weights loaded successfully (strict=True).")

    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)

    # --- Create Dummy Input ---
    print("Creating dummy input...")
    dummy_input_dict = create_dummy_input(config, device)

    # --- Define Input/Output Names ---
    # IMPORTANT: The input names provided here MUST match the keys used
    # inside the model's forward method when accessing the dictionary.
    input_names = list(dummy_input_dict.keys()) # ['pixel_values', 'input_ids', 'attention_mask']
    output_names = ["logits"]

    # --- Ensure Output Directory Exists ---
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # --- Export to ONNX ---
    print(f"Exporting model to ONNX format at {args.output_path} (Opset: {args.opset_version})...")
    try:
        # *** THE FIX IS HERE: Wrap the dummy input dictionary in a tuple ***
        torch.onnx.export(
            model,
            (dummy_input_dict,), # Pass the dictionary wrapped in a tuple
            args.output_path,
            export_params=True,
            opset_version=args.opset_version,
            do_constant_folding=True,
            input_names=input_names,   # Still provide names for dictionary keys for clarity in Netron
            output_names=output_names,
            dynamic_axes={
                'pixel_values': {0: 'batch_size'}, # Reference keys for dynamic axes
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
        print("- Model forward pass expecting a different input format (e.g., tuple vs. dict). Check the input wrapper.")
        print("- Issues with dynamic shapes or control flow within the model.")
        print(f"{'='*80}")
        sys.exit(1)

    print("ONNX export script finished.")
