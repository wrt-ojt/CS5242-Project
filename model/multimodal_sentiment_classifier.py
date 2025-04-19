import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import accuracy_score, classification_report
from tqdm.auto import tqdm # For progress bars

# --- 1. Configuration ---
CONFIG = {
    "data_dir": "data", # Directory containing images and texts
    "label_file": "label.csv", # CSV file with 'id' and 'label' columns
    "clip_model_name": "openai/clip-vit-base-patch32", # Or other CLIP model
    "num_classes": 3, # positive, negative, neutral
    "batch_size": 16, # Adjust based on GPU memory
    "learning_rate_clip": 1e-6, # Smaller LR for pre-trained CLIP
    "learning_rate_head": 1e-4, # Larger LR for custom head
    "num_epochs": 10, # Number of training epochs
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "val_split_ratio": 0.15, # Validation set ratio
    "test_split_ratio": 0.15, # Test set ratio
    "seed": 42, # For reproducible splits/shuffling
    "max_token_length": 77, # Standard CLIP context length
    # --- Ablation Study Flags ---
    "use_cross_attention": True,
    "use_cnn_layer": True,
    "freeze_clip": False # Set to True to freeze CLIP weights initially
}

# Label mapping
label_map = {"negative": 0, "neutral": 1, "positive": 2}
# Inverse mapping for reporting
inv_label_map = {v: k for k, v in label_map.items()}

# --- 2. Dataset and DataLoader ---
class MultimodalBlogDataset(Dataset):
    """Custom Dataset for loading image-text pairs."""
    def __init__(self, data_dir, dataframe, clip_processor, label_map):
        self.data_dir = data_dir
        self.dataframe = dataframe
        self.processor = clip_processor
        self.label_map = label_map
        # Image transformations are handled by CLIPProcessor,
        # but ensure images are loaded correctly (RGB)
        self.image_loader = transforms.Compose([
            transforms.ToTensor() # ToTensor is needed before processor usually
                                  # Processor handles resize and normalize
        ])


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        item_id = row['id']
        label_str = row['label']
        label = self.label_map[label_str]

        # Load Image
        img_path = os.path.join(self.data_dir, f"{item_id}.jpg")
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image file not found {img_path}, returning None.")
            return None # Handle appropriately in collate_fn or dataloader

        # Load Text
        txt_path = os.path.join(self.data_dir, f"{item_id}.txt")
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Warning: Text file not found {txt_path}, returning None.")
            return None # Handle appropriately

        # Preprocessing is done in the training loop / collate_fn
        # Here we just return the raw data + label
        return image, text, label

def collate_fn(batch, processor, device, max_length):
    """Custom collate function to handle preprocessing within the batch."""
    # Filter out None items if any file was not found
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    images, texts, labels = zip(*batch)

    # Process batch using CLIPProcessor
    inputs = processor(
        text=list(texts),
        images=list(images),
        return_tensors="pt",
        padding="max_length", # Pad to max_length
        truncation=True,
        max_length=max_length
    )

    # Move tensors to the correct device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = torch.tensor(labels, dtype=torch.long).to(device)

    return inputs, labels


# --- 3. Model Architecture ---
class MultimodalClassifier(nn.Module):
    """The main model combining CLIP features with a custom fusion head."""
    def __init__(self, clip_model_name, num_classes,
                 use_cross_attention=True, use_cnn_layer=True, freeze_clip=False, device='cpu'):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.use_cnn_layer = use_cnn_layer
        self.device = device

        # Load CLIP model
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)

        # Freeze CLIP weights if specified
        if freeze_clip:
            print("Freezing CLIP model parameters.")
            for param in self.clip_model.parameters():
                param.requires_grad = False
        else:
            print("CLIP model parameters will be fine-tuned.")


        # Get CLIP embedding dimension (projection_dim)
        self.embed_dim = self.clip_model.projection_dim # e.g., 512 or 768

        # --- Fusion Layers ---
        if self.use_cross_attention:
            # MultiheadAttention expects (batch, seq_len, embed_dim) if batch_first=True
            # Our features are (batch, embed_dim), so add seq_len=1
            self.img_to_txt_attention = nn.MultiheadAttention(self.embed_dim, num_heads=8, batch_first=True, dropout=0.1)
            self.txt_to_img_attention = nn.MultiheadAttention(self.embed_dim, num_heads=8, batch_first=True, dropout=0.1)
            fusion_input_dim = self.embed_dim * 4 # img_feat + txt_feat + attended_img + attended_txt
        else:
            fusion_input_dim = self.embed_dim * 2 # img_feat + txt_feat

        # --- CNN Layer (Optional) ---
        # Applying Conv1d on concatenated features of length 1.
        # Kernel size 1 acts like a Linear layer applied independently to each channel.
        # Might not capture "global perception" in the traditional sense here.
        if self.use_cnn_layer:
            self.cnn_out_channels = fusion_input_dim // 2 # Example reduction
            # Input shape for Conv1d: (batch, channels, length)
            # Our concatenated features: (batch, fusion_input_dim)
            # Reshape to: (batch, fusion_input_dim, 1)
            self.conv1d = nn.Conv1d(in_channels=fusion_input_dim,
                                    out_channels=self.cnn_out_channels,
                                    kernel_size=1, # Acts like a linear projection per channel
                                    padding=0)
            self.relu_cnn = nn.ReLU()
            # After Conv1d: (batch, cnn_out_channels, 1) -> Flatten -> (batch, cnn_out_channels)
            classifier_input_dim = self.cnn_out_channels
        else:
            classifier_input_dim = fusion_input_dim # Input dim for MLP if CNN is skipped

        # --- Classifier Head (MLP: Increase then Decrease Dim) ---
        self.classifier_hidden_dim = classifier_input_dim * 2 # "升维"
        self.fc1 = nn.Linear(classifier_input_dim, self.classifier_hidden_dim)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.classifier_hidden_dim, num_classes) # "降维" to num_classes


    def forward(self, inputs):
        # Get CLIP features
        # Note: Use **inputs to unpack dict directly into arguments
        image_features = self.clip_model.get_image_features(pixel_values=inputs['pixel_values'])
        text_features = self.clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        # Features are typically (batch_size, embed_dim)

        # --- Fusion ---
        if self.use_cross_attention:
            # Reshape features for MultiheadAttention: (batch, seq_len=1, embed_dim)
            img_feat_attn = image_features.unsqueeze(1)
            txt_feat_attn = text_features.unsqueeze(1)

            # Image attends to Text (Q=img, K=txt, V=txt)
            attended_img, _ = self.img_to_txt_attention(img_feat_attn, txt_feat_attn, txt_feat_attn)
            attended_img = attended_img.squeeze(1) # Back to (batch, embed_dim)

            # Text attends to Image (Q=txt, K=img, V=img)
            attended_txt, _ = self.txt_to_img_attention(txt_feat_attn, img_feat_attn, img_feat_attn)
            attended_txt = attended_txt.squeeze(1) # Back to (batch, embed_dim)

            # Concatenate all features
            fused_features = torch.cat([image_features, text_features, attended_img, attended_txt], dim=1)
            # Shape: (batch, embed_dim * 4)
        else:
            # Simple concatenation if cross-attention is disabled
            fused_features = torch.cat([image_features, text_features], dim=1)
            # Shape: (batch, embed_dim * 2)

        # --- Optional CNN Layer ---
        if self.use_cnn_layer:
            # Reshape for Conv1d: (batch, channels=fusion_input_dim, length=1)
            cnn_input = fused_features.unsqueeze(2)
            cnn_output = self.conv1d(cnn_input)
            cnn_output = self.relu_cnn(cnn_output)
            # Flatten: (batch, cnn_out_channels, 1) -> (batch, cnn_out_channels)
            classifier_input = cnn_output.squeeze(2)
        else:
            classifier_input = fused_features # Pass concatenated features directly

        # --- Classifier Head ---
        x = self.fc1(classifier_input)
        x = self.relu_fc1(x)
        x = self.dropout1(x)
        logits = self.fc2(x) # Output logits (batch, num_classes)

        return logits

# --- 4. Training and Evaluation Functions ---

def train_epoch(model, dataloader, optimizer, criterion, device, clip_processor, max_length):
    model.train() # Set model to training mode
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        if batch is None: continue # Skip if collate_fn returned None
        inputs, labels = batch

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Store predictions and labels for metric calculation
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

def evaluate_epoch(model, dataloader, criterion, device, clip_processor, max_length):
    model.eval() # Set model to evaluation mode
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad(): # Disable gradient calculations
        for batch in progress_bar:
            if batch is None: continue
            inputs, labels = batch

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Store predictions and labels
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            progress_bar.set_postfix({'loss': loss.item()})


    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=label_map.keys(), zero_division=0)

    return avg_loss, accuracy, report, all_labels, all_preds


# --- 5. Main Execution ---
if __name__ == "__main__":
    print(f"Using device: {CONFIG['device']}")
    torch.manual_seed(CONFIG['seed'])

    # --- Load Data ---
    print("Loading labels...")
    try:
        df = pd.read_csv(CONFIG['label_file'])
        # Basic validation
        if 'id' not in df.columns or 'label' not in df.columns:
             raise ValueError("CSV must contain 'id' and 'label' columns.")
        if not all(label in label_map for label in df['label'].unique()):
            raise ValueError(f"Labels in CSV must be one of {list(label_map.keys())}")
        print(f"Found {len(df)} samples.")
    except FileNotFoundError:
        print(f"Error: Label file not found at {CONFIG['label_file']}")
        exit()
    except ValueError as e:
        print(f"Error: {e}")
        exit()


    # --- Split Data ---
    # Calculate split sizes
    total_size = len(df)
    test_size = int(CONFIG['test_split_ratio'] * total_size)
    val_size = int(CONFIG['val_split_ratio'] * total_size)
    train_size = total_size - val_size - test_size

    print(f"Splitting data: Train={train_size}, Val={val_size}, Test={test_size}")
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        print("Error: Dataset too small for specified split ratios.")
        exit()

    # Perform the split
    train_df, val_df, test_df = random_split(df, [train_size, val_size, test_size],
                                             generator=torch.Generator().manual_seed(CONFIG['seed']))

    # Convert subsets back to DataFrames for easier indexing if needed by Dataset class
    # Note: random_split returns Subset objects. We get the indices and select from the original df.
    train_df = df.iloc[train_df.indices].reset_index(drop=True)
    val_df = df.iloc[val_df.indices].reset_index(drop=True)
    test_df = df.iloc[test_df.indices].reset_index(drop=True)


    # --- Initialize Processor, Dataset, DataLoader ---
    print("Initializing CLIP Processor...")
    clip_processor = CLIPProcessor.from_pretrained(CONFIG['clip_model_name'])

    print("Creating Datasets...")
    train_dataset = MultimodalBlogDataset(CONFIG['data_dir'], train_df, clip_processor, label_map)
    val_dataset = MultimodalBlogDataset(CONFIG['data_dir'], val_df, clip_processor, label_map)
    test_dataset = MultimodalBlogDataset(CONFIG['data_dir'], test_df, clip_processor, label_map)

    print("Creating DataLoaders...")
    # Define the collate function with necessary arguments partially filled
    collate_fn_partial = lambda batch: collate_fn(batch, clip_processor, CONFIG['device'], CONFIG['max_token_length'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn_partial)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn_partial)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn_partial)


    # --- Initialize Model, Loss, Optimizer ---
    print("Initializing Model...")
    # Pass ablation flags here
    model = MultimodalClassifier(
        clip_model_name=CONFIG['clip_model_name'],
        num_classes=CONFIG['num_classes'],
        use_cross_attention=CONFIG["use_cross_attention"],
        use_cnn_layer=CONFIG["use_cnn_layer"],
        freeze_clip=CONFIG["freeze_clip"],
        device=CONFIG['device']
    )
    # model = model.to(CONFIG['device']) # Model parts are moved to device in __init__

    criterion = nn.CrossEntropyLoss()

    # Separate parameters for different learning rates
    clip_params = list(model.clip_model.parameters())
    head_params = []
    if CONFIG["use_cross_attention"]:
        head_params.extend(list(model.img_to_txt_attention.parameters()))
        head_params.extend(list(model.txt_to_img_attention.parameters()))
    if CONFIG["use_cnn_layer"]:
        head_params.extend(list(model.conv1d.parameters()))
    head_params.extend(list(model.fc1.parameters()))
    head_params.extend(list(model.fc2.parameters()))


    optimizer = optim.AdamW([
        {'params': clip_params, 'lr': CONFIG['learning_rate_clip']},
        {'params': head_params, 'lr': CONFIG['learning_rate_head']}
    ])

    # --- Training Loop ---
    print("Starting Training...")
    best_val_accuracy = 0.0
    best_epoch = -1

    for epoch in range(CONFIG['num_epochs']):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['num_epochs']} ---")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, CONFIG['device'], clip_processor, CONFIG['max_token_length'])
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        val_loss, val_acc, val_report, _, _ = evaluate_epoch(model, val_loader, criterion, CONFIG['device'], clip_processor, CONFIG['max_token_length'])
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        print("Validation Classification Report:\n", val_report)

        # Save best model based on validation accuracy
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            # Create a directory to save models if it doesn't exist
            os.makedirs("models", exist_ok=True)
            model_save_path = os.path.join("models", "best_multimodal_model.pth")
            print(f"Validation accuracy improved. Saving model to {model_save_path}")
            torch.save(model.state_dict(), model_save_path)
            # You might want to save optimizer state and epoch number too for resuming training

    print(f"\nTraining finished. Best validation accuracy ({best_val_accuracy:.4f}) achieved at epoch {best_epoch+1}.")

    # --- Final Evaluation on Test Set ---
    print("\n--- Evaluating on Test Set using Best Model ---")
    # Load the best model weights
    best_model_path = os.path.join("models", "best_multimodal_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=CONFIG['device']))
        print("Loaded best model weights for testing.")

        test_loss, test_acc, test_report, test_labels, test_preds = evaluate_epoch(model, test_loader, criterion, CONFIG['device'], clip_processor, CONFIG['max_token_length'])
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print("Test Set Classification Report:\n", test_report)
        # You can further analyze test_labels and test_preds here (e.g., confusion matrix)
    else:
        print("Warning: Best model file not found. Skipping test set evaluation.")

    print("\nDone.")


