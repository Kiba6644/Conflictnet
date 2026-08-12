#!/usr/bin/env python3
"""Connect to Lightning AI and run ConflictNet training on H200."""

import os
import time

from lightning_sdk import User, Teamspace, Studio

# Credentials from environment — NEVER hardcode secrets
STUDIO_NAME = os.environ.get("LIGHTNING_STUDIO", "training-devbox")
TEAMSPACE_NAME = os.environ.get("LIGHTNING_TEAMSPACE", "financial-llm-training-project")

print("Connecting to Lightning AI...")
user = User(name="goelshashwat205")
teamspace = Teamspace(name=TEAMSPACE_NAME, user=user)
studio = Studio(name=STUDIO_NAME, teamspace=teamspace)

if studio.status != "Running":
    print(f"Starting studio '{STUDIO_NAME}'...")
    studio.start()
    print("Waiting for studio to be ready...")
    time.sleep(30)

commands = [
    "cd /teamspace/studios/this_studio",
    "git clone https://github.com/DevodG/ConflictNet.git conflictnet 2>/dev/null || true",
    "cd conflictnet",
    "git pull origin main",
    "pip install -q -r requirements.txt sentencepiece tiktoken kagglehub speechbrain audiomentations",
]

print("Running setup commands...")
for cmd in commands:
    print(f"  $ {cmd}")
    studio.run(cmd)

print("Downloading CREMA-D dataset...")
r = studio.run("python -c \"import kagglehub; print('Downloaded to:', kagglehub.dataset_download('ejlok1/cremad'))\" 2>&1 | tail -3")
print(r)

print("Starting training on H200...")
CREMAD = "/teamspace/studios/this_studio/.cache/kagglehub/datasets/ejlok1/cremad/versions/1"
CMD = (
    f"cd /teamspace/studios/this_studio/conflictnet && "
    f"nohup python scripts/train.py "
    f"--cremad_root {CREMAD} "
    f"--epochs 30 "
    f"--batch_size 32 "
    f"--lr 3e-5 "
    f"--audio_encoder wavlm "
    f"--gradient_accumulation_steps 1 "
    f"--pretrain_epochs 5 "
    f"--amp "
    f"--output_dir /teamspace/studios/this_studio/conflictnet/checkpoints "
    f"--target_f1 0.78 "
    f"--max_retries 2 "
    f"--resume_epochs 10 "
    f"> training.log 2>&1 &"
)
r = studio.run(CMD)
print("Training launched. Check progress with: tail -f training.log")
