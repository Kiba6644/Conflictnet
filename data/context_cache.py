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

from typing import Dict, List, Optional, Tuple

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
        # Store an optional turn index with every embedding.  Training batches
        # are shuffled, so append order alone can otherwise leak future dialogue
        # turns into the current sample's context.
        self._cache: Dict[str, List[Tuple[Optional[int], torch.Tensor]]] = {}

    def get_context(self, conv_id: str, before_turn: Optional[int] = None) -> Optional[torch.Tensor]:
        """Get context history for a conversation.

        Returns:
            ``(T_ctx, embed_dim)`` tensor of past turns, or None if
            the conversation has no history yet.
        """
        if conv_id not in self._cache:
            return None
        history = self._cache[conv_id]
        if before_turn is not None:
            # Indexed turns are only valid if strictly earlier. Unindexed
            # histories are retained for datasets that do not expose turns.
            history = [(turn, embed) for turn, embed in history if turn is None or turn < before_turn]
        if not history:
            return None
        if all(turn is not None for turn, _ in history):
            history = sorted(history, key=lambda item: item[0])
        history = history[-self.max_turns:]
        return torch.cat([embed for _, embed in history], dim=0)

    def get_batch_context(
        self,
        conv_ids: List[str],
        embed_dim: int = 256,
        turn_indices: Optional[List[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Get padded context for a batch of conversations.

        Returns:
            - ``context_embeds: (B, T_pad, embed_dim)`` — zero-padded context
              sequences. Samples with no history get all-zero context.
            - ``context_padding: (B, T_pad)`` bool mask (True = padded).
            - ``context_conversations: (B,)`` same conv_ids.
        """
        B = len(conv_ids)
        contexts: List[Optional[torch.Tensor]] = []
        for i, conv_id in enumerate(conv_ids):
            current_turn = turn_indices[i] if turn_indices is not None else None
            contexts.append(self.get_context(conv_id, before_turn=current_turn))
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

    def update(self, conv_id: str, turn_embed: torch.Tensor, turn_index: Optional[int] = None):
        """Append a single turn embedding to conversation history.

        Args:
            conv_id: Conversation identifier.
            turn_embed: ``(embed_dim,)`` or ``(1, embed_dim)`` tensor.
        """
        fe = turn_embed.detach()
        if fe.dim() == 1:
            fe = fe.unsqueeze(0)
        history = self._cache.setdefault(conv_id, [])
        if turn_index is not None:
            # A padded DistributedSampler can repeat examples. Replace rather
            # than duplicate an already-seen turn.
            history[:] = [(turn, embed) for turn, embed in history if turn != turn_index]
        history.append((turn_index, fe))
        if len(history) > self.max_turns * 2:
            if all(turn is not None for turn, _ in history):
                history.sort(key=lambda item: item[0])
            del history[:-self.max_turns]

    def batch_update(
        self,
        conv_ids: List[str],
        turn_embeds: torch.Tensor,
        turn_indices: Optional[List[int]] = None,
    ):
        """Update cache for all samples in a batch in chronological turn order.

        Args:
            conv_ids: Conversation identifiers for each sample.
            turn_embeds: ``(B, embed_dim)`` — one per sample.
            turn_indices: Optional list of turn indices. If provided, samples
                are updated in turn order so the cache reflects the dialogue sequence.
        """
        if turn_indices is not None:
            order = sorted(range(len(conv_ids)), key=lambda i: turn_indices[i])
        else:
            order = range(len(conv_ids))
        for i in order:
            turn_index = turn_indices[i] if turn_indices is not None else None
            self.update(conv_ids[i], turn_embeds[i], turn_index=turn_index)

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
