"""GPT-4o text-only baseline for ConflictNet.

Provides a text-only ceiling for comparison against ConflictNet's
multimodal predictions. Classifies conflict type from transcript alone.

Usage:
    python scripts/llm_baseline.py --test_json /path/to/test.json --output results.json
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert in detecting emotion in spoken language.

You will be given a transcript of an utterance. Your task is to:
1. Determine if the utterance contains emotional conflict (anger, disgust, or fear are conflict emotions)
2. Classify the primary emotion from: anger, disgust, fear, happiness, neutral, sadness
3. Rate the severity from 0.0 (none) to 1.0 (extreme)

Respond in valid JSON only, no other text:
{
  "conflict": true/false,
  "types": {"anger": 0/1, "disgust": 0/1, "fear": 0/1, "happiness": 0/1, "neutral": 0/1, "sadness": 0/1},
  "severity": 0.0-1.0,
  "reasoning": "brief explanation"
}"""


def classify_utterance(
    transcript: str,
    client,
    model: str = "gpt-4o",
    max_tokens: int = 256,
    max_retries: int = 3,
) -> Optional[Dict]:
    """Classify a single utterance using GPT-4o."""
    for attempt in range(max_retries):
        try:
            create_fn = getattr(client.chat.completions, "create")
            response = create_fn(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f'Utterance: "{transcript}"'},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            message = response.choices[0].message
            refusal = getattr(message, "refusal", None)
            if refusal and isinstance(refusal, str):
                raise Exception(f"Model refused request: {refusal}")
            result = json.loads(message.content)
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                logger.warning(f"[LLM] Retry {attempt+1} for: {transcript[:50]}...")
            else:
                logger.error(f"[LLM] Failed after {max_retries} retries: {e}")
                return None


def run_llm_baseline(
    test_items: List[Dict],
    output_path: str,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    max_samples: int = -1,
) -> Dict:
    """Run LLM baseline on a list of test items.

    Args:
        test_items: List of dicts with keys 'text', 'conflict_binary', 'conflict_type_labels'.
        output_path: Path to save JSON results.
        model: OpenAI model ID.
        api_key: API key (defaults to OPENAI_API_KEY env var).
        max_samples: Limit number of samples (-1 = all).

    Returns:
        Dict with predictions and metrics.
    """
    try:
        import openai  # type: ignore
    except ImportError:
        raise ImportError("pip install openai to use the LLM baseline")

    client = openai.OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    if max_samples > 0:
        test_items = test_items[:max_samples]

    results = []
    for item in test_items:
        pred = classify_utterance(item["text"], client, model)
        if pred:
            results.append({
                "text": item["text"],
                "true_conflict": item.get("conflict_binary", 0),
                "true_types": item.get("conflict_type_labels", [0, 0, 0, 0, 1, 0]),  # default: neutral
                "pred_conflict": int(pred.get("conflict", False)),
                "pred_types": [
                    pred.get("types", {}).get("anger", 0),
                    pred.get("types", {}).get("disgust", 0),
                    pred.get("types", {}).get("fear", 0),
                    pred.get("types", {}).get("happiness", 0),
                    pred.get("types", {}).get("neutral", 0),
                    pred.get("types", {}).get("sadness", 0),
                ],
                "pred_severity": pred.get("severity", 0.0),
                "reasoning": pred.get("reasoning", ""),
            })

    # Compute metrics
    import numpy as np
    from evaluation.metrics import compute_all_metrics

    if results:
        probs = np.array([[r["pred_types"][i] for i in range(6)] for r in results], dtype=float)
        labels = np.array([r["true_types"] for r in results], dtype=float)
        metrics = compute_all_metrics(probs, labels)
        metrics["model"] = model
    else:
        metrics = {}

    output = {"results": results, "metrics": metrics}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"[LLM Baseline] Saved {len(results)} results → {output_path}")
    return output
