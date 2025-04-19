# src/dataset.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import random

try:
    from config import CONFIG
except ImportError:
    # Handle case where script is run directly or config is not in PYTHONPATH
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.config import CONFIG

class PreprocessedMultimodalDataset(Dataset):
    """Dataset to load preprocessed tensors saved by preprocess.py."""
    def __init__(self, split, config):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            config (dict): Configuration dictionary.
        """
        self.data_dir = os.path.join(config['preprocessed_data_dir'], split)
        self.config = config
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Preprocessed data directory not found: {self.data_dir}. Run preprocess.py first.")

        self.file_paths = glob(os.path.join(self.data_dir, "*.pt"))
        if not self.file_paths:
             raise FileNotFoundError(f"No '.pt' files found in {self.data_dir}. Preprocessing might have failed or directory is empty.")

        print(f"Found {len(self.file_paths)} preprocessed samples for split '{split}'.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            data = torch.load(file_path, map_location='cpu') # Load to CPU first
            return data
        except Exception as e:
            print(f"Error loading preprocessed file {file_path}: {e}. Returning None.")
            # The collate function should handle None values
            return None

def collate_fn(batch):
    """
    Custom collate function for preprocessed data.
    Handles potential None values from dataset loading errors.
    """
    # Filter out None items
    batch = [item for item in batch if item is not None]
    if not batch:
        return None # Return None if the whole batch failed

    keys = batch[0].keys()
    collated_batch = {key: [] for key in keys}

    for item in batch:
        for key in keys:
            collated_batch[key].append(item[key])

    # Stack tensors for relevant keys
    try:
        collated_batch['pixel_values'] = torch.stack(collated_batch['pixel_values'])
        collated_batch['input_ids'] = torch.stack(collated_batch['input_ids'])
        collated_batch['attention_mask'] = torch.stack(collated_batch['attention_mask'])
        collated_batch['label'] = torch.stack(collated_batch['label'])
        # 'original_id' might be int or string, keep as list or handle as needed
    except Exception as e:
        print(f"Error during batch collation (stacking tensors): {e}")
        return None # Indicate error

    return collated_batch

def create_dataloaders(config):
    """Creates dataloaders for train, validation, and test splits."""
    train_dataset = PreprocessedMultimodalDataset('train', config)
    val_dataset = PreprocessedMultimodalDataset('val', config)
    test_dataset = PreprocessedMultimodalDataset('test', config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True if config['device'] == 'cuda' else False # Improve transfer speed if using CUDA
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True if config['device'] == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True if config['device'] == 'cuda' else False
    )
    print(f"DataLoaders created with batch_size={config['batch_size']} and num_workers={config['num_workers']}")
    return train_loader, val_loader, test_loader
