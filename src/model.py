# src/model.py
import torch
import torch.nn as nn
from transformers import CLIPModel
import sys, os

try:
    from config import CONFIG # Assuming config is accessible
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.config import CONFIG


class MultimodalClassifier(nn.Module):
    """
    Enhanced multimodal classifier with configurable fusion and head.
    Supports single-modality inference as well.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config['device']
        self.modality = config['modality']

        # --- Load CLIP ---
        print(f"Loading CLIP model: {config['clip_model_name']}")
        self.clip_model = CLIPModel.from_pretrained(config['clip_model_name'])
        # Move CLIP to device early if not freezing all parts instantly
        self.clip_model.to(self.device)

        # --- Freeze CLIP ---
        if config['freeze_clip']:
            print("Freezing CLIP model parameters.")
            for param in self.clip_model.parameters():
                param.requires_grad = False
        else:
            print("CLIP model parameters will be fine-tuned.")
            # Ensure model is on device if fine-tuning
            # self.clip_model.to(self.device) # Already moved above

        self.embed_dim = self.clip_model.projection_dim # Output dim of CLIP features

        # --- Optional Projection Layers (Before Fusion) ---
        self.projection_dim = config.get('projection_dim', None) # Use .get for safety
        if self.projection_dim:
            print(f"Adding projection layers to dimension: {self.projection_dim}")
            self.image_projection = nn.Linear(self.embed_dim, self.projection_dim)
            self.text_projection = nn.Linear(self.embed_dim, self.projection_dim)
            self.current_embed_dim = self.projection_dim # Dimension after projection
            self.proj_activation = nn.ReLU() # Example activation
        else:
            self.current_embed_dim = self.embed_dim # Dimension is original CLIP output dim

        # --- Fusion Logic ---
        fusion_input_dim = 0
        if self.modality == 'multimodal':
            if config['use_cross_attention']:
                print("Using Cross-Attention fusion.")
                self.img_to_txt_attention = nn.MultiheadAttention(
                    self.current_embed_dim, num_heads=config['num_attention_heads'],
                    batch_first=True, dropout=config['dropout_attention']
                )
                self.txt_to_img_attention = nn.MultiheadAttention(
                    self.current_embed_dim, num_heads=config['num_attention_heads'],
                    batch_first=True, dropout=config['dropout_attention']
                )
                # Final concatenated dim: original_img + original_text + attended_img + attended_text
                fusion_input_dim = self.current_embed_dim * 4
            else:
                print("Using simple Concatenation fusion.")
                # Final concatenated dim: original_img + original_text
                fusion_input_dim = self.current_embed_dim * 2
        elif self.modality == 'image':
            print("Using Image modality only.")
            fusion_input_dim = self.current_embed_dim
        elif self.modality == 'text':
            print("Using Text modality only.")
            fusion_input_dim = self.current_embed_dim
        else:
            raise ValueError(f"Invalid modality specified: {self.modality}")

        # --- Optional CNN Layer ---
        # Note: Conv1D with kernel=1 is just a Linear layer applied channel-wise.
        # Consider if this is truly beneficial or if MLP is sufficient.
        if config['use_cnn_layer'] and self.modality == 'multimodal': # Only makes sense for fused features
            print("Using optional Conv1D layer after fusion.")
            cnn_out_channels = int(fusion_input_dim * config['cnn_out_channels_ratio'])
            self.conv1d = nn.Conv1d(
                in_channels=fusion_input_dim,
                out_channels=cnn_out_channels,
                kernel_size=1, padding=0
            )
            self.relu_cnn = nn.ReLU()
            classifier_input_dim = cnn_out_channels
        else:
            classifier_input_dim = fusion_input_dim # Input to the final classifier head

        # --- Dynamic Classifier Head (MLP) ---
        print(f"Building Classifier Head. Input dim: {classifier_input_dim}")
        mlp_layers = []
        hidden_dims = config.get('classifier_hidden_layers', []) # Default to empty list
        input_dim = classifier_input_dim
        if hidden_dims:
            for i, hidden_dim in enumerate(hidden_dims):
                mlp_layers.append(nn.Linear(input_dim, hidden_dim))
                mlp_layers.append(nn.ReLU())
                mlp_layers.append(nn.Dropout(config['dropout_mlp']))
                input_dim = hidden_dim # Input dim for next layer
                print(f"  MLP Layer {i+1}: Linear({mlp_layers[-3].in_features}, {mlp_layers[-3].out_features}), ReLU, Dropout({config['dropout_mlp']})")
        # Final layer to num_classes
        mlp_layers.append(nn.Linear(input_dim, config['num_classes']))
        print(f"  MLP Final Layer: Linear({mlp_layers[-1].in_features}, {mlp_layers[-1].out_features})")
        self.classifier_head = nn.Sequential(*mlp_layers)

        # Move relevant layers to device (Attention, CNN, MLP, Projections)
        self.to(self.device)
        # Note: CLIP model was already moved

    def forward(self, batch):
        # Assumes batch contains preprocessed tensors moved to the correct device
        pixel_values = batch['pixel_values']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        # --- Get CLIP Features ---
        image_features_clip = None
        text_features_clip = None

        if self.modality in ['multimodal', 'image']:
            image_features_clip = self.clip_model.get_image_features(pixel_values=pixel_values)
        if self.modality in ['multimodal', 'text']:
            text_features_clip = self.clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

        # --- Apply Optional Projection ---
        image_features = image_features_clip
        text_features = text_features_clip
        if self.projection_dim:
            if image_features is not None:
                image_features = self.proj_activation(self.image_projection(image_features))
            if text_features is not None:
                text_features = self.proj_activation(self.text_projection(text_features))

        # --- Fusion / Modality Selection ---
        fused_features = None
        if self.modality == 'multimodal':
            if self.config['use_cross_attention']:
                img_feat_attn = image_features.unsqueeze(1) # (B, 1, D)
                txt_feat_attn = text_features.unsqueeze(1) # (B, 1, D)
                attended_img, _ = self.img_to_txt_attention(img_feat_attn, txt_feat_attn, txt_feat_attn)
                attended_txt, _ = self.txt_to_img_attention(txt_feat_attn, img_feat_attn, img_feat_attn)
                fused_features = torch.cat([image_features, text_features, attended_img.squeeze(1), attended_txt.squeeze(1)], dim=1)
            else:
                fused_features = torch.cat([image_features, text_features], dim=1)
        elif self.modality == 'image':
            fused_features = image_features
        elif self.modality == 'text':
            fused_features = text_features

        # --- Optional CNN ---
        if self.config['use_cnn_layer'] and self.modality == 'multimodal':
             # Reshape for Conv1d: (batch, channels=fusion_input_dim, length=1)
            cnn_input = fused_features.unsqueeze(2)
            cnn_output = self.relu_cnn(self.conv1d(cnn_input))
            classifier_input = cnn_output.squeeze(2) # Flatten
        else:
            classifier_input = fused_features

        # --- Classifier Head ---
        logits = self.classifier_head(classifier_input)

        return logits
