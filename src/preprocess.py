# src/preprocess.py
import os
import pandas as pd
from PIL import Image
import torch
from transformers import CLIPProcessor
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import argparse
import sys

# Ensure src directory is in path for config import if running directly
# Or rely on PYTHONPATH / running via run_experiment.py
try:
    from config import CONFIG
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.config import CONFIG


def process_and_save(df, split_name, config, processor):
    """Loads raw data, processes using CLIPProcessor, and saves tensors."""
    raw_data_dir = config['raw_data_dir']
    output_dir = os.path.join(config['preprocessed_data_dir'], split_name)
    os.makedirs(output_dir, exist_ok=True)
    label_map = config['label_map']
    max_length = config['max_token_length']
    processed_count = 0
    skipped_count = 0

    print(f"Processing split: {split_name}...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Preprocessing {split_name}"):
        item_id_int = row['ID'] # Assuming ID column name
        item_id = str(item_id_int) # Ensure string for filename formatting
        label_str = row['label'] # Assuming label column name

        if label_str not in label_map:
            print(f"Warning: Skipping item {item_id} due to unknown label '{label_str}'")
            skipped_count += 1
            continue

        label = label_map[label_str]

        # --- Load Image ---
        img_path = os.path.join(raw_data_dir, f"{item_id}.jpg")
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image file not found {img_path}, skipping item {item_id}.")
            skipped_count += 1
            continue
        except Exception as e:
            print(f"Warning: Error loading image {img_path}: {e}, skipping item {item_id}.")
            skipped_count += 1
            continue

        # --- Load Text ---
        txt_path = os.path.join(raw_data_dir, f"{item_id}.txt")
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Warning: Text file not found {txt_path}, skipping item {item_id}.")
            skipped_count += 1
            continue
        except Exception as e:
             print(f"Warning: Error loading text {txt_path}: {e}, skipping item {item_id}.")
             skipped_count += 1
             continue

        # --- Process using CLIPProcessor ---
        try:
            # Process image and text separately to handle potential individual errors
            # Note: padding/truncation applied here
            inputs = processor(
                text=[text],        # Process one item at a time
                images=[image],     # Process one item at a time
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
            # Squeeze batch dimension added by processor when processing single items
            processed_data = {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'label': torch.tensor(label, dtype=torch.long),
                'original_id': item_id_int # Keep original ID if needed later
            }
        except Exception as e:
            print(f"Warning: Error processing item {item_id} with CLIPProcessor: {e}, skipping.")
            skipped_count += 1
            continue

        # --- Save processed data ---
        save_path = os.path.join(output_dir, f"{item_id}.pt")
        torch.save(processed_data, save_path)
        processed_count += 1

    print(f"Finished {split_name}. Processed: {processed_count}, Skipped: {skipped_count}")

def main(config):
    """Main function to orchestrate preprocessing."""
    print("--- Starting Data Preprocessing ---")

    # Check if preprocessing already done and force_preprocess is False
    train_dir = os.path.join(config['preprocessed_data_dir'], 'train')
    if os.path.exists(train_dir) and not config['force_preprocess'] and len(os.listdir(train_dir)) > 0 :
        print(f"Preprocessed data found at {config['preprocessed_data_dir']}. Skipping preprocessing.")
        print("Set 'force_preprocess: True' in config to rerun.")
        return

    # Load labels
    print(f"Loading labels from {config['label_file']}...")
    try:
        df = pd.read_csv(config['label_file']).dropna(how='all')
        # Basic validation (adapt column names 'ID', 'label' if different)
        df['ID'] = df['ID'].astype(int)
        df['class'] = df['class'].astype(int)
        if 'ID' not in df.columns or 'label' not in df.columns:
             raise ValueError("CSV must contain 'ID' and 'label' columns.")
        # Optional: Convert ID early if needed, handle potential errors
        # df['ID'] = df['ID'].astype(int)
        print(f"Found {len(df)} samples in label file.")
    except FileNotFoundError:
        print(f"Error: Label file not found at {config['label_file']}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error reading label file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred loading the label file: {e}")
        sys.exit(1)

    # --- Split Data ---
    print("Splitting data...")
    val_test_size = config['val_split_ratio'] + config['test_split_ratio']
    if val_test_size >= 1.0:
        print("Error: Sum of validation and test split ratios must be less than 1.")
        sys.exit(1)

    # Adjust test size relative to the remaining data after validation split
    relative_test_size = config['test_split_ratio'] / (1.0 - config['val_split_ratio'])

    try:
        # Split into train and temp (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=val_test_size,
            random_state=config['seed'],
            stratify=df['label'] # Stratify if labels are imbalanced
        )
        # Split temp into val and test
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_size,
            random_state=config['seed'],
            stratify=temp_df['label'] # Stratify if labels are imbalanced
        )
    except Exception as e:
        print(f"Error during data splitting: {e}. Check split ratios and data.")
        # Might happen if a label class has too few samples for stratification
        print("Attempting split without stratification...")
        try:
            train_df, temp_df = train_test_split(df, test_size=val_test_size, random_state=config['seed'])
            val_df, test_df = train_test_split(temp_df, test_size=relative_test_size, random_state=config['seed'])
        except Exception as e_nostrat:
            print(f"Error during non-stratified split: {e_nostrat}.")
            sys.exit(1)


    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        print("Error: One or more splits have zero samples. Check data and split ratios.")
        sys.exit(1)

    # --- Initialize CLIP Processor ---
    print(f"Initializing CLIP Processor: {config['clip_model_name']}...")
    try:
      processor = CLIPProcessor.from_pretrained(config['clip_model_name'])
    except Exception as e:
      print(f"Error initializing CLIP Processor: {e}")
      sys.exit(1)

    # --- Process and Save Each Split ---
    process_and_save(train_df, 'train', config, processor)
    process_and_save(val_df, 'val', config, processor)
    process_and_save(test_df, 'test', config, processor)

    print("--- Data Preprocessing Finished ---")

if __name__ == "__main__":
    # Allows running preprocess.py directly
    parser = argparse.ArgumentParser(description="Preprocess multimodal data.")
    # Add arguments to override config values if needed, e.g.:
    # parser.add_argument('--force', action='store_true', help="Force reprocessing even if data exists.")
    args = parser.parse_args()
    # Example of using arg: if args.force: CONFIG['force_preprocess'] = True
    main(CONFIG)
