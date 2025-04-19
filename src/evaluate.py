# src/evaluate.py
import torch
import os
import logging
from .train import evaluate_epoch # Reuse evaluation logic
from .utils import save_results

def test_model(config, model, test_loader, criterion):
    """Evaluates the final model on the test set."""
    device = config['device']
    best_model_path = os.path.join(config['experiment_output_dir'], "best_model.pth")
    results_path = os.path.join(config['experiment_output_dir'], "test_results.json")

    logging.info("\n--- Evaluating on Test Set ---")
    if not os.path.exists(best_model_path):
        logging.error(f"Best model file not found at {best_model_path}. Cannot perform testing.")
        return None

    # Load the best model weights
    try:
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logging.info(f"Loaded best model weights from {best_model_path} for testing.")
    except Exception as e:
        logging.error(f"Error loading best model weights: {e}")
        return None

    model.to(device) # Ensure model is on the correct device
    test_loss, test_acc, test_report_str, test_report_dict, _, _ = evaluate_epoch(
        model, test_loader, criterion, device, config
    )

    logging.info(f"\nTest Loss: {test_loss:.4f}")
    logging.info(f"Test Accuracy: {test_acc:.4f}")
    logging.info(f"Test Set Classification Report:\n{test_report_str}")

    # Save results
    results = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_classification_report_dict": test_report_dict,
        "test_classification_report_str": test_report_str,
    }
    save_results(results, results_path)

    return results
