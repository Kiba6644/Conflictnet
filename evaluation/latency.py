"""Latency benchmarking for ConflictNet."""
import time
import torch
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def benchmark_latency(
    model: torch.nn.Module,
    dummy_batch: Dict[str, torch.Tensor],
    n_warmup: int = 10,
    n_iters: int = 100,
    device: str = "cuda",
) -> Dict[str, float]:
    """Benchmark model inference latency.

    Returns:
        Dict with avg_ms, std_ms, throughput (samples/sec), p95_ms, p99_ms.
    """
    model.eval()
    model.to(device)
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in dummy_batch.items()}

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            model(**batch)

    # Timed runs
    latencies = []
    for _ in range(n_iters):
        start = time.perf_counter()
        with torch.no_grad():
            model(**batch)
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    latencies = sorted(latencies)
    avg_ms = sum(latencies) / len(latencies)
    std_ms = (sum((lat - avg_ms) ** 2 for lat in latencies) / len(latencies)) ** 0.5
    batch_size = dummy_batch["audio"].size(0)

    return {
        "avg_ms": round(avg_ms, 2),
        "std_ms": round(std_ms, 2),
        "p95_ms": round(latencies[int(0.95 * len(latencies))], 2),
        "p99_ms": round(latencies[int(0.99 * len(latencies))], 2),
        "throughput": round(batch_size / (avg_ms / 1000), 1),
        "batch_size": batch_size,
        "n_iters": n_iters,
    }
