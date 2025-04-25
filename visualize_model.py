# visualize_model.py
import torch
import os
import argparse
from torchinfo import summary as torchinfo_summary

# Make src importable
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.config import CONFIG # Import default config
from src.model import MultimodalClassifier
# Potentially import override_config logic from run_experiment if needed
# from run_experiment import override_config # Or copy the function here

# --- Dummy Input Creation ---
def create_dummy_input(config, device):
    """Creates a dummy input batch matching model expectations."""
    batch_size = 2 # Use a small batch size for visualization
    max_len = config['max_token_length']
    # CLIP image processor typically outputs (B, C, H, W), e.g., (B, 3, 224, 224)
    # Use a placeholder size if exact size isn't critical for structure summary
    # but check CLIPProcessor documentation or an actual processed batch for accuracy
    dummy_pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_input_ids = torch.randint(0, config.get('vocab_size', 30522), (batch_size, max_len)).to(device) # Use actual vocab size if known
    dummy_attention_mask = torch.ones(batch_size, max_len, dtype=torch.long).to(device)

    dummy_batch = {
        'pixel_values': dummy_pixel_values,
        'input_ids': dummy_input_ids,
        'attention_mask': dummy_attention_mask
        # Label is not needed for forward pass structure/graph
    }
    return dummy_batch

# --- Main Visualization Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Multimodal Model")
    # Add arguments to override config keys affecting model structure if needed
    parser.add_argument('--config_overrides', type=str, help="JSON string to override config keys, e.g., '{\"modality\":\"image\"}'")
    parser.add_argument('--save_graph_path', type=str, default="output/model_graph", help="Path prefix to save torchviz graph (e.g., output/model_graph -> .gv, .gv.pdf)")
    args = parser.parse_args()

    config = CONFIG.copy() # Start with default config

    # --- Apply Overrides (Optional) ---
    # Simple override example, enhance as needed
    if args.config_overrides:
        import json
        try:
            overrides = json.loads(args.config_overrides)
            config.update(overrides)
            print("Applied config overrides:", overrides)
        except json.JSONDecodeError:
            print("Warning: Could not parse config_overrides JSON string.")

    device = torch.device(config['device'])

    # --- Instantiate Model ---
    print("Initializing model for visualization...")
    # Ensure model uses the potentially overridden config
    model = MultimodalClassifier(config).to(device)
    model.eval() # Set to eval mode

    # --- Create Dummy Input ---
    dummy_input = create_dummy_input(config, device)
    print("Created dummy input batch.")

    # --- 1. torchinfo Summary ---
    print("\n" + "="*80)
    print("Model Summary (torchinfo):")
    print("="*80)
    try:
        # Define input structure for torchinfo
        # Providing input_data is often more reliable for complex inputs (dicts)
        summary = torchinfo_summary(
            model,
            input_data=dummy_input,
            # Or by input_shapes if input wasn't a dict:
            # input_size=[(config['batch_size'], 3, 224, 224), (config['batch_size'], config['max_token_length'])],
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
            depth=5, # Adjust depth as needed
            verbose=0 # 0 for summary, 1 for detailed print
        )
        # summary object already prints when created with verbose=0
        # print(summary)
    except Exception as e:
        print(f"Could not generate torchinfo summary: {e}")

    # # --- 2. torchviz Graph ---
    # print("\n" + "="*80)
    # print("Generating Computation Graph (torchviz)...")
    # print("="*80)
    # try:
    #     import torchviz
    #     # Perform a forward pass to get output
    #     output = model(dummy_input)
    #     # Create the graph
    #     dot = torchviz.make_dot(output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    #     # Save the graph (creates .gv and .gv.pdf/.png etc.)
    #     # Make sure the directory exists
    #     os.makedirs(os.path.dirname(args.save_graph_path), exist_ok=True)
    #     dot.render(args.save_graph_path, format='pdf', view=False) # Save as PDF
    #     print(f"Saved torchviz graph files starting with: {args.save_graph_path}")
    #     print(f"(Look for {args.save_graph_path}.pdf)")
    # except ImportError:
    #     print("torchviz or graphviz not found. Skipping graph generation.")
    #     print("Install with: pip install torchviz graphviz")
    #     print("(You might also need to install graphviz system-wide)")
    # except Exception as e:
    #     print(f"Could not generate torchviz graph: {e}")
    #     # Common issue: Graphviz executable not found in PATH

    # print("\nVisualization script finished.")

    # --- 2. torchviz Graph ---
    print("\n" + "="*80)
    print("Generating Computation Graph (torchviz)...")
    print("="*80)
    try:
        import torchviz
        # Perform a forward pass to get output
        # Ensure model is in eval mode if dropout/batchnorm layers affect graph structure differently
        model.eval()
        output = model(dummy_input)

        # Create the graph - simplified options
        # Set params=None, show_saved=False, show_attrs=False
        dot = torchviz.make_dot(
            output,
            params=None, # Try removing parameter nodes
            show_attrs=False, # Try removing node attributes
            show_saved=False # Try removing saved tensor nodes
        )

        # Save the graph (creates .gv and .gv.pdf/.png etc.)
        # Make sure the directory exists
        os.makedirs(os.path.dirname(args.save_graph_path), exist_ok=True)
        dot.render(args.save_graph_path, format='pdf', view=False) # Save as PDF
        print(f"Saved potentially simplified torchviz graph files starting with: {args.save_graph_path}")
        print(f"(Look for {args.save_graph_path}.pdf)")

    except ImportError:
        print("torchviz or graphviz not found. Skipping graph generation.")
        print("Install with: pip install torchviz graphviz")
        print("(You might also need to install graphviz system-wide)")
    except Exception as e:
        print(f"Could not generate torchviz graph: {e}")
        # Common issue: Graphviz executable not found in PATH
