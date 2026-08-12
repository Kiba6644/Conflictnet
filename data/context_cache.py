"""Context cache for dialogue turn history.

Stores per-conversation fused embeddings so that prior turns can be
used as context for cross-attention injection and temporal context.

Usage in trainer::

    cache = ContextCache(max_turns=8, device=device)
    ...
    # Before forward: populate context_embeds from cache
    conv_ids = batch["conversation_ids"]
    context_embeds = cache.get_batch_context(conv_ids)

    # After forward: update cache with current turn embeddings
    cache.batch_update(conv_ids, context_pooled, turn_indices)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch


class ContextCache:
    """Stores and retrieves dialogue turn embeddings per conversation.

    For each conversation, maintains a rolling window of past turn
    fused embeddings. Used to provide ``context_embeds`` for the
    cross-attention injector and temporal context module.

    Args:
        max_turns: Maximum number of past turns to keep per conversation.
        device: Device for stored tensors.
    """

    def __init__(self, max_turns: int = 8, device: str = "cpu"):
        self.max_turns = max_turns
        self.device = device
        self._cache: Dict[str, torch.Tensor] = {}

    def get_context(self, conv_id: str) -> Optional[torch.Tensor]:
        """Get context history for a conversation.

        Returns:
            ``(T_ctx, embed_dim)`` tensor of past turns, or None if
            the conversation has no history yet.
        """
        if conv_id not in self._cache:
            return None
        ctx = self._cache[conv_id]
        if ctx.size(0) > self.max_turns:
            ctx = ctx[-self.max_turns:]
        return ctx

    def get_batch_context(
        self, conv_ids: List[str], embed_dim: int = 256
    ) -> tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Get padded context for a batch of conversations.

        Returns:
            - ``context_embeds: (B, T_pad, embed_dim)`` — zero-padded context
              sequences. Samples with no history get all-zero context.
            - ``context_padding: (B, T_pad)`` bool mask (True = padded).
            - ``context_conversations: (B,)`` same conv_ids.
        """
        B = len(conv_ids)
        contexts: List[Optional[torch.Tensor]] = [
            self.get_context(cid) for cid in conv_ids
        ]
        max_len = max(
            (ctx.size(0) for ctx in contexts if ctx is not None), default=0
        )
        if max_len == 0:
            embeds = torch.zeros(B, 1, embed_dim, device=self.device)
            padding = torch.ones(B, 1, dtype=torch.bool, device=self.device)
            return embeds, padding, conv_ids

        T = max(1, min(max_len, self.max_turns))
        embeds = torch.zeros(B, T, embed_dim, device=self.device)
        padding = torch.ones(B, T, dtype=torch.bool, device=self.device)

        for i, ctx in enumerate(contexts):
            if ctx is not None:
                ctx = ctx.to(self.device)
                n = min(ctx.size(0), T)
                embeds[i, -n:] = ctx[-n:]
                padding[i, -n:] = False

        return embeds, padding, conv_ids

    def update(self, conv_id: str, turn_embed: torch.Tensor):
        """Append a single turn embedding to conversation history.

        Args:
            conv_id: Conversation identifier.
            turn_embed: ``(embed_dim,)`` or ``(1, embed_dim)`` tensor.
        """
        fe = turn_embed.detach()
        if fe.dim() == 1:
            fe = fe.unsqueeze(0)
        if conv_id in self._cache:
            self._cache[conv_id] = torch.cat([self._cache[conv_id], fe], dim=0)
        else:
            self._cache[conv_id] = fe
        if self._cache[conv_id].size(0) > self.max_turns * 2:
            self._cache[conv_id] = self._cache[conv_id][-self.max_turns:]

    def batch_update(
        self,
        conv_ids: List[str],
        turn_embeds: torch.Tensor,
    ):
        """Update cache for all samples in a batch.

        Args:
            conv_ids: Conversation identifiers for each sample.
            turn_embeds: ``(B, embed_dim)`` — one per sample.
        """
        for i, cid in enumerate(conv_ids):
            self.update(cid, turn_embeds[i])

    def clear(self, conv_id: Optional[str] = None):
        """Clear cache for one or all conversations."""
        if conv_id is None:
            self._cache.clear()
        elif conv_id in self._cache:
            del self._cache[conv_id]

    def to(self, device: str) -> ContextCache:
        self.device = device
        return self

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return f"ContextCache(max_turns={self.max_turns}, conversations={len(self._cache)})"
