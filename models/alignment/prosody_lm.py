import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

class ProsodyLM(nn.Module):
    """Prosody LM Auxiliary Task (small GPT-2 scale transformer for text -> prosody prediction)."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 256, num_layers: int = 4, num_heads: int = 4):
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=embed_dim,
            n_layer=num_layers,
            n_head=num_heads,
            n_positions=1024
        )
        self.plm = GPT2Model(config)
        
        # Predict prosody vector from token representation
        # Prosody is represented as a sequence of features per token, e.g. 8-dim divergence vector
        self.prosody_head = nn.Linear(embed_dim, 8)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, actual_prosody: torch.Tensor = None):
        """
        Args:
            input_ids: (B, L) textual inputs
            attention_mask: (B, L)
            actual_prosody: (B, L, 8) target prosody features
            
        Returns:
            predicted_prosody: (B, L, 8)
            residual: (B, L, 8) difference between predicted and actual, which acts as conflict signal
        """
        outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, L, D)
        
        predicted_prosody = self.prosody_head(hidden_states)
        
        residual = None
        if actual_prosody is not None:
            residual = actual_prosody - predicted_prosody
            
        return predicted_prosody, residual
