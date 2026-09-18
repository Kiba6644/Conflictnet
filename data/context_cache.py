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
        # Store turn index, embedding, and speaker role with every entry.
        self._cache: Dict[str, List[Tuple[Optional[int], torch.Tensor, int]]] = {}

    def get_context(
        self, conv_id: str, before_turn: Optional[int] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get context history for a conversation.

        Returns:
            ``(embeds, speaker_roles)`` tuple:
            - ``embeds: (T_ctx, embed_dim)`` tensor of past turns, or None.
            - ``speaker_roles: (T_ctx,)`` tensor of speaker roles, or None.
        """
        if conv_id not in self._cache:
            return None, None
        history = self._cache[conv_id]
        if before_turn is not None:
            history = [item for item in history if item[0] is None or item[0] < before_turn]
        if not history:
            return None, None
        if all(item[0] is not None for item in history):
            history = sorted(history, key=lambda item: item[0])
        history = history[-self.max_turns:]
        embeds = torch.cat([item[1] for item in history], dim=0)
        roles = torch.tensor([item[2] for item in history], dtype=torch.long)
        return embeds, roles

    def get_batch_context(
        self,
        conv_ids: List[str],
        embed_dim: int = 256,
        turn_indices: Optional[List[int]] = None,
        return_roles: bool = False,
    ):
        """Get padded context for a batch of conversations.

        Args:
            conv_ids: List of conversation IDs.
            embed_dim: Dimension of turn embeddings.
            turn_indices: Optional current turn indices for causal slicing.
            return_roles: If True, returns 4-tuple including context speaker roles.
                          If False, returns 3-tuple for backwards compatibility.

        Returns:
            If return_roles is False:
                - ``context_embeds: (B, T_pad, embed_dim)``
                - ``context_padding: (B, T_pad)`` bool mask (True = padded)
                - ``context_conversations: (B,)`` conv_ids
            If return_roles is True:
                - ``context_embeds: (B, T_pad, embed_dim)``
                - ``context_padding: (B, T_pad)`` bool mask
                - ``context_speaker_roles: (B, T_pad)`` long tensor of speaker roles
                - ``context_conversations: (B,)`` conv_ids
        """
        B = len(conv_ids)
        contexts: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = []
        for i, conv_id in enumerate(conv_ids):
            current_turn = turn_indices[i] if turn_indices is not None else None
            contexts.append(self.get_context(conv_id, before_turn=current_turn))
        max_len = max(
            (ctx_e.size(0) for ctx_e, _ in contexts if ctx_e is not None), default=0
        )
        if max_len == 0:
            embeds = torch.zeros(B, 1, embed_dim, device=self.device)
            padding = torch.ones(B, 1, dtype=torch.bool, device=self.device)
            roles = torch.zeros(B, 1, dtype=torch.long, device=self.device)
            if return_roles:
                return embeds, padding, roles, conv_ids
            return embeds, padding, conv_ids

        T = max(1, min(max_len, self.max_turns))
        embeds = torch.zeros(B, T, embed_dim, device=self.device)
        padding = torch.ones(B, T, dtype=torch.bool, device=self.device)
        roles = torch.zeros(B, T, dtype=torch.long, device=self.device)

        for i, (ctx_e, ctx_r) in enumerate(contexts):
            if ctx_e is not None and ctx_r is not None:
                ctx_e = ctx_e.to(self.device)
                ctx_r = ctx_r.to(self.device)
                n = min(ctx_e.size(0), T)
                embeds[i, -n:] = ctx_e[-n:]
                roles[i, -n:] = ctx_r[-n:]
                padding[i, -n:] = False

        if return_roles:
            return embeds, padding, roles, conv_ids
        return embeds, padding, conv_ids

    def update(
        self,
        conv_id: str,
        turn_embed: torch.Tensor,
        turn_index: Optional[int] = None,
        speaker_role: int = 0,
    ):
        """Append a single turn embedding and speaker role to conversation history.

        Args:
            conv_id: Conversation identifier.
            turn_embed: ``(embed_dim,)`` or ``(1, embed_dim)`` tensor.
            turn_index: Optional turn index.
            speaker_role: Integer speaker role in [0, 15].
        """
        fe = turn_embed.detach()
        if fe.dim() == 1:
            fe = fe.unsqueeze(0)
        history = self._cache.setdefault(conv_id, [])
        if turn_index is not None:
            history[:] = [item for item in history if item[0] != turn_index]
        history.append((turn_index, fe, int(speaker_role)))
        if len(history) > self.max_turns * 2:
            if all(item[0] is not None for item in history):
                history.sort(key=lambda item: item[0])
            del history[:-self.max_turns]

    def batch_update(
        self,
        conv_ids: List[str],
        turn_embeds: torch.Tensor,
        turn_indices: Optional[List[int]] = None,
        speaker_roles: Optional[Any] = None,
    ):
        """Update cache for all samples in a batch in chronological turn order.

        Args:
            conv_ids: Conversation identifiers for each sample.
            turn_embeds: ``(B, embed_dim)`` — one per sample.
            turn_indices: Optional list of turn indices.
            speaker_roles: Optional ``(B,)`` tensor or list of speaker role ints.
        """
        if turn_indices is not None:
            order = sorted(range(len(conv_ids)), key=lambda i: turn_indices[i])
        else:
            order = range(len(conv_ids))
        for i in order:
            turn_index = turn_indices[i] if turn_indices is not None else None
            if speaker_roles is not None:
                role = speaker_roles[i].item() if isinstance(speaker_roles, torch.Tensor) else int(speaker_roles[i])
            else:
                role = 0
            self.update(conv_ids[i], turn_embeds[i], turn_index=turn_index, speaker_role=role)

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
