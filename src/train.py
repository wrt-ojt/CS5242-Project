# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report
import time
import os
import logging

# Assuming utils.py is in the same directory or path is handled
from .utils import save_checkpoint, load_checkpoint

def evaluate_epoch(model, dataloader, criterion, device, config):
    """Evaluates the model on a given dataset split."""
    model.eval() # Set model to evaluation mode
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad(): # Disable gradient calculations
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            if batch is None:
                logging.warning("Skipping None batch during evaluation.")
                continue

            # Move batch items to device (collate_fn might handle some)
            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor) and k != 'label'}
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Store predictions and labels
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0

    # Generate classification report
    try:
        report_dict = classification_report(all_labels, all_preds,
                                             target_names=config['inv_label_map'].values(),
                                             output_dict=True, zero_division=0)
        report_str = classification_report(all_labels, all_preds,
                                           target_names=config['inv_label_map'].values(),
                                           zero_division=0)
    except Exception as e:
        logging.error(f"Error generating classification report: {e}")
        report_dict = {}
        report_str = "Error generating report."

    return avg_loss, accuracy, report_str, report_dict, all_labels, all_preds


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Trains the model for one epoch."""
    model.train() # Set model to training mode
    total_loss = 0
    batch_count = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        if batch is None:
            logging.warning("Skipping None batch during training.")
            continue

        # Move batch items to device
        inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor) and k != 'label'}
        labels = batch['label'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        # Optional: Gradient clipping can help stabilize training
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    return avg_loss

def train_model(config, model, train_loader, val_loader, optimizer, criterion):
    """Main training loop with validation and early stopping."""
    device = config['device']
    num_epochs = config['num_epochs']
    patience = config['early_stopping_patience']
    output_dir = config['experiment_output_dir']
    best_model_path = os.path.join(output_dir, "best_model.pth")
    last_checkpoint_path = os.path.join(output_dir, "last_checkpoint.pth")

    best_val_accuracy = 0.0
    epochs_no_improve = 0
    start_epoch = 0

    # --- Optional: Resume Training ---
    # if os.path.exists(last_checkpoint_path):
    #     logging.info(f"Resuming training from {last_checkpoint_path}")
    #     model, optimizer, start_epoch = load_checkpoint(model, optimizer, last_checkpoint_path, device)
    #     # Load best accuracy achieved so far if needed (might need to save it in checkpoint)
    #     start_epoch += 1 # Start from the next epoch

    logging.info("--- Starting Training ---")
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        logging.info(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        logging.info(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")

        val_loss, val_acc, val_report_str, _, _, _ = evaluate_epoch(model, val_loader, criterion, device, config)
        logging.info(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        logging.info(f"Validation Classification Report (Epoch {epoch+1}):\n{val_report_str}")

        epoch_duration = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch+1} duration: {epoch_duration:.2f} seconds")

        # --- Checkpoint Saving & Early Stopping ---
        # Save last checkpoint
        # save_checkpoint(model, optimizer, epoch, last_checkpoint_path)

        # Check if validation accuracy improved
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), best_model_path) # Save only state_dict for best model
            logging.info(f"Validation accuracy improved to {best_val_accuracy:.4f}. Saving best model to {best_model_path}")
        else:
            epochs_no_improve += 1
            logging.info(f"Validation accuracy did not improve. Current best: {best_val_accuracy:.4f}. Epochs without improvement: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            logging.info(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    logging.info(f"\nTraining finished. Best validation accuracy: {best_val_accuracy:.4f}")
    return best_model_path
