"""Average the last N checkpoint weights for a free performance boost.
Uses stochastic weight averaging (SWA) principle: the average of recent
checkpoints often lies in a flatter loss basin → better generalization.
"""
import argparse
from pathlib import Path
from safetensors.torch import load_file, save_file
import torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint_dir")
    p.add_argument("--n", type=int, default=5, help="Average last N checkpoints")
    p.add_argument("--output", type=str, default="averaged_model.safetensors")
    args = p.parse_args()

    ckpts = sorted(Path(args.checkpoint_dir).glob("*.safetensors"))
    ckpts = ckpts[-args.n:]
    print(f"Averaging {len(ckpts)} checkpoints: {[c.name for c in ckpts]}")

    avg = {}
    for ckpt in ckpts:
        state = load_file(str(ckpt))
        for k, v in state.items():
            if k not in avg:
                avg[k] = v.float() / len(ckpts)
            else:
                avg[k] += v.float() / len(ckpts)

    avg = {k: v.half() for k, v in avg.items()}
    save_file(avg, args.output)
    print(f"Saved averaged checkpoint to {args.output}")

if __name__ == "__main__":
    main()
