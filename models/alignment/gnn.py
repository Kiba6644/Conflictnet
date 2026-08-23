import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiPartyGNN(nn.Module):
    """Graph Neural Network for Multi-Party Conversation Modelling.
    
    Nodes = speaker turns
    Edges = temporal connections and same-speaker connections
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        # Edge type weights: temporal vs same-speaker
        self.edge_weights = nn.Parameter(torch.ones(2))
        
    def forward(self, node_features: torch.Tensor, temporal_edges: torch.Tensor, speaker_edges: torch.Tensor):
        """
        Args:
            node_features: (N, D) embeddings for N utterances in a dialogue window
            temporal_edges: (2, E_temp) standard sequence adjacency
            speaker_edges: (2, E_speak) same-speaker connections
            
        Returns:
            updated_features: (N, D) context-aware embeddings
        """
        h = node_features
        
        # We can implement a simplified message passing
        for i, layer in enumerate(self.layers):
            # Message passing
            msg_temporal = self._aggregate(h, temporal_edges)
            msg_speaker = self._aggregate(h, speaker_edges)
            
            # Combine messages
            combined_msg = self.edge_weights[0] * msg_temporal + self.edge_weights[1] * msg_speaker
            
            # Update
            h = layer(h + combined_msg)
            if i < self.num_layers - 1:
                h = F.relu(h)
                
        return h
        
    def _aggregate(self, h: torch.Tensor, edges: torch.Tensor):
        if edges.size(1) == 0:
            return torch.zeros_like(h)
            
        src, dst = edges
        N = h.size(0)
        
        out = torch.zeros_like(h)
        # scatter_add is better but let's do simple index_add
        out.index_add_(0, dst, h[src])
        
        # Normalize by degree
        degree = torch.zeros(N, device=h.device)
        ones = torch.ones(edges.size(1), device=h.device)
        degree.index_add_(0, dst, ones)
        
        degree = degree.clamp(min=1).unsqueeze(-1)
        return out / degree
