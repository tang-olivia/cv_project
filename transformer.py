import torch
import torch.nn as nn
import torch.nn.functional as F

class JointEmbedTransformer(nn.Module):
    def __init__(self, image_dim=1536, sequence_dim=1536, model_dim=512, num_layers=2, num_heads=8, dropout=0.1):
        super(JointEmbedTransformer, self).__init__()
        self.model_dim = model_dim
        
        self.image_projector = nn.Linear(image_dim, model_dim)
        self.sequence_projector = nn.Linear(sequence_dim, model_dim)
        
        self.cls_token = nn.Parameter(torch.randn(1, model_dim))
        
        self.positional_encoding = nn.Parameter(torch.randn(3, model_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, image_emb, sequence_emb):
        batch_size = image_emb.size(0)

        image_emb = F.normalize(image_emb, dim=1)
        sequence_emb = F.normalize(sequence_emb, dim=1)
        
        image_proj = self.image_projector(image_emb)
        sequence_proj = self.sequence_projector(sequence_emb)
        
        cls_tokens = self.cls_token.expand(batch_size, -1)
        
        input_sequence = torch.stack([cls_tokens, image_proj, sequence_proj], dim=1)
        
        input_sequence = input_sequence + self.positional_encoding
        
        output = self.transformer(input_sequence)
        
        joint_emb = output[:, 0, :]
        return joint_emb
