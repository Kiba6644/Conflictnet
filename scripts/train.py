"""CLI: train ConflictNet v2.

Usage:
    python scripts/train.py --config configs/default.yaml \
        --iemocap_root /data/iemocap \
        --mustard_root  /data/mustard \
        --output_dir checkpoints/run1 \
        --pretrain_epochs 5 \
        --epochs 30
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

# Fix for macOS OpenMP multiple initialization error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Fix for DeBERTa v2 "fabs" TorchScript compilation bug
os.environ["PYTORCH_JIT"] = "0"
# Suppress Hugging Face and ModelScope download progress bars to keep Kaggle logs clean
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["MS_DISABLE_PROGRESS_BAR"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fix for Kaggle dual-T4 NCCL deadlocks during DDP broadcast


# Use file_system sharing strategy to avoid /dev/shm exhaustion.
# Default num_workers=0 means no shared memory is needed at all.
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Disable Memory-Efficient and Flash Attention backends globally.
# On Kaggle's dual-T4 (Turing architecture) GPUs, PyTorch's scaled_dot_product_attention 
# (which nn.MultiheadAttention uses) can enter infinite loops or cause Segmentation Faults 
# when query_len=1 and key_len>1 (e.g. cross-modal attention with context).
# Math backend is 100% stable and fast enough for our sequence lengths.
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

from torch.utils.data import DataLoader, ConcatDataset
try:
    from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:
    def record(fn):
        return fn

# Add project root to sys.path so 'data', 'models' etc. can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[FlushHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train ConflictNet v2")
    p.add_argument("--iemocap_root", type=str, default=None)
    p.add_argument("--mustard_root", type=str, default=None)
    p.add_argument("--mustard_wav_dir", type=str, default="utterances_final", help="Path to MUStARD wav files")
    p.add_argument("--cremad_root", type=str, default=None, help="CREMA-D dataset root")
    p.add_argument("--meld_root", type=str, default=None, help="MELD dataset root")
    p.add_argument("--meld_max_samples", type=int, default=None,
                   help="Legacy arg, use meld_max_train_samples instead")
    p.add_argument("--meld_max_train_samples", type=int, default=None,
                   help="Cap MELD train split to this many samples (stratified)")
    p.add_argument("--meld_max_val_samples", type=int, default=None,
                   help="Cap MELD val split to this many samples (stratified)")
    p.add_argument("--meld_textgrid_root", type=str, default=None,
                   help="Path to directory containing MFA TextGrid files for MELD word alignment")
    p.add_argument("--mustard_textgrid_root", type=str, default=None,
                   help="Path to directory containing MFA TextGrid files for MUStARD word alignment")
    p.add_argument("--musan_path", type=str, default=None, help="MUSAN corpus for noise augmentation")
    p.add_argument("--output_dir", type=str, default="checkpoints")
    p.add_argument("--audio_encoder", type=str, default="wavlm_weighted",
                   choices=["emotion2vec", "wavlm", "wavlm_weighted", "wav2vec2", "whisper", "dual"])
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable gradient checkpointing on WavLM to reduce VRAM usage")
    p.add_argument("--unfreeze_audio_layers", type=int, default=16,
                   help="Number of final WavLM transformer layers to unfreeze for fine-tuning. "
                        "0=fully frozen (layer_weights only), 16=recommended for 2xT4, 24=full fine-tune. "
                        "Ignored for non-WavLM encoders.")
    p.add_argument("--audio_encoder_path", type=str, default=None,
                   help="Path to local audio encoder directory to bypass ModelScope")
    p.add_argument("--text_encoder_path", type=str, default=None,
                   help="Path to local text encoder directory to bypass HuggingFace")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--amp", action="store_true", default=True,
                   help="Enable automatic mixed precision (enabled by default)")
    p.add_argument("--no_amp", action="store_true",
                   help="Disable automatic mixed precision (fp16/bf16) training")
    p.add_argument("--compile", action="store_true", help="Enable torch.compile (overrides config)")
    p.add_argument("--pretrain_epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=150)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_speaker_norm", action="store_true")
    p.add_argument("--no_temporal", action="store_true",
                   help="Disable Transformer temporal context module")
    p.add_argument("--no_cross_attn_injection", action="store_true",
                   help="Disable cross-attention injection from temporal context into audio+text")
    p.add_argument("--no_speaker_adaptive_threshold", action="store_true",
                   help="Disable speaker-adaptive divergence threshold (use fixed threshold)")
    p.add_argument("--no_baseline_subtract", action="store_true",
                   help="Disable baseline-subtract prosody normalisation (use z-score instead)")
    p.add_argument("--no_word_divergence", action="store_true")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32,
                   help="LoRA alpha scaling (recommended: equal to lora_r for scaling=1.0)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--early_stop_patience", type=int, default=15,
                   help="Stop if val F1 does not improve for this many epochs (default 15 to allow "
                        "WavLM fine-tuning plateaus to resolve)")
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--prosody_stats", type=str, default=None,
                   help="Path to .pt file from compute_prosody_stats.py with per-utterance z-scores")
    p.add_argument("--tokenizer_path", type=str, default=None,
                   help="Path to local tokenizer directory (avoids HuggingFace download)")
    p.add_argument("--target_f1", type=float, default=0.0,
                   help="Target val F1. If not met after training, resume with halved LR (0 = disable)")
    p.add_argument("--max_retries", type=int, default=0,
                   help="Max times to continue training if below target_f1")
    p.add_argument("--resume_epochs", type=int, default=10,
                   help="Additional epochs per retry")
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Label smoothing epsilon for conflict type BCE loss (0 = disabled)")
    p.add_argument("--use_adaptive_router", action="store_true",
                   help="Enable learned modality router alpha gate (text vs audio weighting)")
    p.add_argument("--router_entropy_reg", type=float, default=0.01,
                   help="Entropy regularization for modality router (prevents collapse)")
    p.add_argument("--augment_p", type=float, default=0.5,
                   help="Probability of applying augmentation per sample (0=off, 1=always)")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader worker processes. Default 0 (single process). Set >0 if running in an environment with sufficient shared memory.")
    p.add_argument("--pt_dir", type=str, default=None,
                   help="Path to precomputed features directory (sets CONFLICTNET_PT_DIR automatically)")
    return p.parse_args(argv)


@record
def main(args=None):
    if args is None:
        args = parse_args(argv=None)

    if args.pt_dir:
        os.environ["CONFLICTNET_PT_DIR"] = str(Path(args.pt_dir).resolve())
        logger.info(f"[ConflictNet] Set CONFLICTNET_PT_DIR={os.environ['CONFLICTNET_PT_DIR']}")

    # torchrun workers are not guaranteed to retain the notebook shell's
    # working directory. Resolve output paths relative to this repository once
    # so every rank shares manifests and checkpoint files.
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    args.output_dir = str(output_dir.resolve())
    
    def fix_kaggle_path(p: str) -> str:
        if p and "/datasets/nith27/" in p:
            fixed = p.replace("/datasets/nith27/", "/")
            if os.path.exists(fixed):
                return fixed
        return p

    if args.audio_encoder_path:
        args.audio_encoder_path = fix_kaggle_path(args.audio_encoder_path)
        os.environ["CONFLICTNET_EMOTION2VEC_PATH"] = args.audio_encoder_path
        os.environ["CONFLICTNET_WAVLM_PATH"] = args.audio_encoder_path
    if args.text_encoder_path:
        args.text_encoder_path = fix_kaggle_path(args.text_encoder_path)
        os.environ["CONFLICTNET_DEBERTA_PATH"] = args.text_encoder_path
    if args.tokenizer_path:
        args.tokenizer_path = fix_kaggle_path(args.tokenizer_path)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    if local_rank != -1:
        # Fix for Kaggle dual-T4 NCCL deadlocks
        # Kaggle's T4 GPUs do not support P2P over PCIe properly. This causes
        # any DDP collective operations (like barrier or broadcast) to hang forever.
        
        
        
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Define this before dataset construction: MELD/MUStARD dataset constructors
    # initialise a Hugging Face tokenizer, so starting the model warm-up later is
    # too late to protect a fresh cache from concurrent torchrun workers.
    from models.conflictnet import ConflictNet

    if (getattr(args, "pt_dir", None) or os.environ.get("CONFLICTNET_PT_DIR")) and args.audio_encoder != "precomputed":
        logger.info("Using precomputed audio embeddings: setting audio_encoder to 'precomputed' to save ~2.5GB VRAM.")
        args.audio_encoder = "precomputed"

    def _build_model():
        return ConflictNet(
            audio_encoder_name=args.audio_encoder,
            embed_dim=args.embed_dim,
            use_speaker_norm=not args.no_speaker_norm,
            use_temporal=not args.no_temporal,
            use_cross_attn_injection=not args.no_cross_attn_injection,
            use_speaker_adaptive_threshold=not args.no_speaker_adaptive_threshold,
            use_baseline_subtract=not args.no_baseline_subtract,
            use_word_divergence=not args.no_word_divergence,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            label_smoothing=args.label_smoothing,
            gradient_checkpointing=args.gradient_checkpointing,
            unfreeze_audio_layers=args.unfreeze_audio_layers,
            use_adaptive_router=args.use_adaptive_router,
            router_entropy_reg=args.router_entropy_reg,
        )

    is_ddp_run = local_rank != -1
    prewarmed_model = None
    
    if is_ddp_run:
        # Hugging Face and SpeechBrain use disk caches, but they are not a
        # transaction across all of the files a model needs.  In particular,
        # letting every rank build a dataset first can race on the tokenizer
        # before the old, model-only barrier below is reached.  Build every
        # pretrained component once on rank zero before *any* dataset or model
        # constructor runs in another worker.
        
        # IMPORTANT: We MUST do this BEFORE init_process_group!
        # If DDP is initialized, SpeechBrain's `from_hparams` will internally execute 
        # DDP barriers. Because we wrap this in `if local_rank == 0`, Rank 1 would 
        # skip those barriers, permanently desynchronizing the NCCL queue.
        if local_rank == 0:
            logger.info("[DDP] Rank 0 pre-warming pretrained model and tokenizer caches...")
            rng_state = torch.get_rng_state()
            cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            
            # Pre-warm model
            # Keep this instance for rank 0's eventual training model.  Building
            # it a second time can take a different optional fallback path than
            # the first build (notably PEFT / pretrained encoder initialisation),
            # while every other rank builds only once from the warmed cache.
            prewarmed_model = _build_model()
            
            audio_b = getattr(prewarmed_model.audio_encoder, "_backend", "auto")
            text_b = getattr(prewarmed_model.text_encoder, "_backend", "auto")
            lora_b = getattr(prewarmed_model.text_encoder, "_lora_backend", "auto")
            
            # Pre-warm tokenizer (used by all datasets)
            from transformers import AutoTokenizer
            logger.info("[DDP] Rank 0 downloading/caching tokenizer...")
            tok_name = args.tokenizer_path or "microsoft/deberta-v3-large"
            _ = AutoTokenizer.from_pretrained(tok_name)
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_states)
            torch.set_rng_state(rng_state)
            logger.info("[DDP] Pre-warm complete — releasing the remaining ranks.")
            
        # Now that we've safely pre-warmed everything on Rank 0 without 
        # SpeechBrain messing up the DDP queue, we can safely initialize DDP!
        import datetime
        torch.distributed.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(seconds=300)
        )
            
        if local_rank == 0:
            logger.info("[DDP] Rank 0 waiting at first barrier...")
        torch.distributed.barrier()
        if local_rank == 0:
            logger.info("[DDP] First barrier passed. Broadcasting tensor...")
        
        # Broadcast backend decisions using a simple tensor to avoid NCCL object serialization bugs
        backend_state = torch.zeros(3, dtype=torch.long, device=args.device)
        AUDIO_MAP = {"auto": 0, "spectrogram": 1, "pretrained": 2, "funasr": 3, "fallback_wavlm": 4}
        TEXT_MAP = {"auto": 0, "fallback": 1, "pretrained": 2}
        LORA_MAP = {"auto": 0, "frozen": 1, "peft": 2}
        
        if local_rank == 0:
            backend_state[0] = AUDIO_MAP.get(audio_b, 0)
            backend_state[1] = TEXT_MAP.get(text_b, 0)
            backend_state[2] = LORA_MAP.get(lora_b, 0)
            
        torch.distributed.broadcast(backend_state, src=0)
        
        if local_rank == 0:
            logger.info("[DDP] Broadcast complete. Setting environment variables...")
            
        AUDIO_INV = {v: k for k, v in AUDIO_MAP.items()}
        TEXT_INV = {v: k for k, v in TEXT_MAP.items()}
        LORA_INV = {v: k for k, v in LORA_MAP.items()}
        
        audio_final = AUDIO_INV.get(backend_state[0].item(), "auto")
        text_final = TEXT_INV.get(backend_state[1].item(), "auto")
        lora_final = LORA_INV.get(backend_state[2].item(), "auto")
        
        os.environ["CONFLICTNET_WAVLM_BACKEND"] = audio_final
        os.environ["CONFLICTNET_EMOTION2VEC_BACKEND"] = audio_final
        os.environ["CONFLICTNET_TEXT_BACKEND"] = text_final
        os.environ["CONFLICTNET_LORA_BACKEND"] = lora_final

        if local_rank == 0:
            logger.info("[DDP] Variables set. Waiting at second barrier...")
        torch.distributed.barrier()
        if local_rank == 0:
            logger.info("[DDP] Second barrier passed. Starting dataset building...")

    # --- Build datasets ---
    from data.datasets import (
        IEMOCAPDataset, MUStARDDataset, CREMADDataset, MELDDataset,
        make_collate_fn,
    )

    # Create augmentation-aware collate functions via closure (fork-safe)
    from data.augmentation import AudioAugmentor

    # Load pre-computed prosody z-scores if available
    prosody_lookup = None
    prosody_path = args.prosody_stats
    if not prosody_path:
        default_prosody = repo_root / "prosody_stats.zscores.json"
        if default_prosody.exists():
            prosody_path = str(default_prosody)
            logger.info(f"Auto-detected default prosody stats: {prosody_path}")

    if prosody_path:
        prosody_path = Path(prosody_path)
        zscores_p = prosody_path.parent / f"{prosody_path.stem}.zscores.json"
        if not zscores_p.exists():
            zscores_p = prosody_path.with_suffix(".zscores.json")
        if not zscores_p.exists() and prosody_path.name.endswith(".json"):
            zscores_p = prosody_path
            
        if zscores_p.exists():
            try:
                with open(zscores_p) as f:
                    raw = json.load(f)
                prosody_lookup = {k: torch.tensor(v) for k, v in raw.items()}
                if not prosody_lookup:
                    prosody_lookup = None
                else:
                    logger.info(f"[Prosody] loaded {len(prosody_lookup)} z-score entries from {zscores_p}")
            except Exception as e:
                logger.warning(f"[Prosody] failed to load {zscores_p}: {e}")
        else:
            logger.warning(f"[Prosody] lookup file not found: {zscores_p}")

    augmentor = AudioAugmentor(
        sample_rate=16000,
        musan_path=getattr(args, "musan_path", None),
        speed_perturb=False,  # Disabled: desynchronizes prosody_z and MFA word timestamps
        additive_noise=True,
        time_mask=True,
        p=getattr(args, "augment_p", 0.5),
    )
    # Both collate fns use the SAME prosody lookup (z-scores computed from
    # training-data-only speaker statistics by compute_prosody_stats.py).
    # This is CORRECT — val utterances get z-scores based on training speaker
    # statistics, preventing data leakage across splits.
    train_collate = make_collate_fn(augmentor=augmentor, prosody_lookup=prosody_lookup)
    val_collate = make_collate_fn(prosody_lookup=prosody_lookup)

    train_datasets = []
    val_datasets = []

    if args.iemocap_root:
        logger.info(f"[Rank {local_rank}] Loading IEMOCAP dataset...")
        # Leave-one-session-out: sessions 1-4 train, 5 val
        train_datasets.append(IEMOCAPDataset(args.iemocap_root, sessions=[1, 2, 3, 4]))
        val_datasets.append(IEMOCAPDataset(args.iemocap_root, sessions=[5]))
        logger.info(f"[Rank {local_rank}] IEMOCAP loaded.")

    if args.mustard_root:
        logger.info(f"[Rank {local_rank}] Loading MUStARD dataset...")
        mustard_kwargs = {}
        if args.mustard_textgrid_root:
            mustard_kwargs["textgrid_root"] = args.mustard_textgrid_root
        if args.tokenizer_path:
            mustard_kwargs["tokenizer_name"] = args.tokenizer_path
            
        train_datasets.append(MUStARDDataset(
            root=args.mustard_root,
            wav_dir=args.mustard_wav_dir,
            split="train",
            **mustard_kwargs
        ))
        val_datasets.append(MUStARDDataset(
            root=args.mustard_root,
            wav_dir=args.mustard_wav_dir,
            split="val",
            **mustard_kwargs
        ))
        logger.info(f"[Rank {local_rank}] MUStARD loaded.")

    if args.cremad_root:
        logger.info(f"[Rank {local_rank}] Loading CREMA-D dataset...")
        tok_kwargs = {"tokenizer_name": args.tokenizer_path} if args.tokenizer_path else {}
        train_datasets.append(CREMADDataset(args.cremad_root, split="train", **tok_kwargs))
        val_datasets.append(CREMADDataset(args.cremad_root, split="val", **tok_kwargs))
        logger.info(f"[Rank {local_rank}] CREMA-D loaded.")

    if args.meld_root:
        logger.info(f"[Rank {local_rank}] Loading MELD dataset...")
        train_max = args.meld_max_train_samples or args.meld_max_samples
        val_max = args.meld_max_val_samples or args.meld_max_samples
        
        meld_train_kwargs = {"max_samples": train_max} if train_max else {}
        meld_val_kwargs = {"max_samples": val_max} if val_max else {}
        if args.meld_textgrid_root:
            meld_train_kwargs["textgrid_root"] = args.meld_textgrid_root
            meld_val_kwargs["textgrid_root"] = args.meld_textgrid_root
        
        if args.tokenizer_path:
            meld_train_kwargs["tokenizer_name"] = args.tokenizer_path
            meld_val_kwargs["tokenizer_name"] = args.tokenizer_path
            
        logger.info(f"[Rank {local_rank}] Initializing MELD train...")
        train_datasets.append(MELDDataset(args.meld_root, split="train", **meld_train_kwargs))
        logger.info(f"[Rank {local_rank}] Initializing MELD val...")
        val_datasets.append(MELDDataset(args.meld_root, split="val", **meld_val_kwargs))
        logger.info(f"[Rank {local_rank}] MELD loaded.")

    logger.info(f"[Rank {local_rank}] Finished loading all dataset components.")

    # In-line Feature Extraction for actual used files
    # Only perform feature pre-extraction if we are using frozen features (unfreeze_audio_layers == 0)
    # and CONFLICTNET_PT_DIR is explicitly set. If unfreeze_audio_layers > 0, we MUST train end-to-end on raw audio!
    pt_dir = os.environ.get("CONFLICTNET_PT_DIR", "")
    if args.unfreeze_audio_layers == 0 and pt_dir:
        if local_rank in [-1, 0]:
            def collect_paths(ds_list):
                paths = []
                for ds in ds_list:
                    for item in ds.items:
                        if "wav_path" in item and item["wav_path"] is not None:
                            paths.append(Path(item["wav_path"]))
                return paths

            logger.info("[Rank 0] Collecting audio paths for required features...")
            required_audio_files = collect_paths(train_datasets) + collect_paths(val_datasets)
            required_audio_files = list(set(required_audio_files))
            
            from scripts.extract_features import extract_features_for_files
            logger.info(f"[Rank 0] Starting in-line feature extraction into {pt_dir}...")
            extract_features_for_files(required_audio_files, pt_dir, batch_size=args.batch_size or 16, audio_encoder_name=args.audio_encoder)
            logger.info("[Rank 0] In-line feature extraction complete.")
    else:
        # Clear CONFLICTNET_PT_DIR to ensure load_audio loads raw audio waveforms for end-to-end WavLM training
        os.environ["CONFLICTNET_PT_DIR"] = ""
        logger.info(f"[Rank {local_rank}] End-to-end audio training active (unfreeze_audio_layers={args.unfreeze_audio_layers}). Raw waveforms will be passed directly to {args.audio_encoder}.")
        
    if is_ddp_run:
        import torch.distributed as dist
        dist.barrier()  # Wait for rank 0 to finish check before proceeding

    if not train_datasets:
        raise ValueError("Provide at least one of --iemocap_root, --mustard_root, --cremad_root, or --meld_root")

    train_set = ConcatDataset(train_datasets)
    val_set = ConcatDataset(val_datasets)

    from data.samplers import DialogueDistributedBatchSampler

    # Set up training batch sampler
    if is_ddp_run:
        import torch.distributed as dist
        train_batch_sampler = DialogueDistributedBatchSampler(
            train_set, 
            batch_size=args.batch_size or 16, 
            num_replicas=dist.get_world_size(), 
            rank=dist.get_rank(),
            shuffle=True
        )
    else:
        train_batch_sampler = DialogueDistributedBatchSampler(
            train_set, 
            batch_size=args.batch_size or 16, 
            num_replicas=1, 
            rank=0,
            shuffle=True
        )

    # Set up validation batch sampler
    val_batch_sampler = DialogueDistributedBatchSampler(
        val_set, 
        batch_size=args.batch_size or 16, 
        num_replicas=1, 
        rank=0,
        shuffle=False
    )

    optimal_workers = args.num_workers
    _use_persistent = optimal_workers > 0
    _prefetch = 2 if optimal_workers > 0 else None
    _pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_sampler=train_batch_sampler,
        num_workers=optimal_workers,
        collate_fn=train_collate,
        pin_memory=_pin_memory,
        persistent_workers=_use_persistent,
        prefetch_factor=_prefetch,
    )
    
    val_loader = DataLoader(
        val_set,
        batch_sampler=val_batch_sampler,
        num_workers=min(optimal_workers, 2) if optimal_workers > 0 else 0,
        collate_fn=val_collate,
        pin_memory=_pin_memory,
        persistent_workers=_use_persistent,
        prefetch_factor=_prefetch,
    )

    logger.info(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")

    # --- Build model (DDP-safe: rank 0 first) ---
    # In multi-GPU DDP runs torchrun spawns all worker processes simultaneously.
    # If every rank tries to download HuggingFace / SpeechBrain weights at the
    # same time, one rank can get a partial/corrupt file and fall back to a
    # dummy encoder (4 trainable tensors vs 102), which the DDP consistency
    # check then correctly rejects with RuntimeError.
    #
    # The cache was warmed before dataset construction.  Keep this second
    # rank-zero-first construction as a safeguard for non-cache state such as
    # SpeechBrain's local savedir.
    if is_ddp_run:
        if local_rank == 0:
            logger.info("[DDP] Rank 0 building model and warming HF/SpeechBrain cache...")
            model = prewarmed_model if prewarmed_model is not None else _build_model()
            logger.info("[DDP] Rank 0 done — releasing barrier for other ranks.")
        torch.distributed.barrier()  # non-zero ranks wait until rank 0 finishes all downloads
        if local_rank != 0:
            logger.info(f"[DDP] Rank {local_rank} building model from warm cache...")
            model = _build_model()
        torch.distributed.barrier()  # all ranks confirm they've finished building
    else:
        model = _build_model()

    param_counts = model.count_parameters()
    total_trainable = sum(v["trainable"] for v in param_counts.values())
    logger.info(f"Total trainable parameters: {total_trainable:,}")
    if is_ddp_run:
        trainable_tensors = {
            name: sum(1 for p in module.parameters() if p.requires_grad)
            for name, module in model.named_children()
        }
        audio = model.audio_encoder
        logger.info(
            f"[DDP] Rank {local_rank} trainable tensors by component: {trainable_tensors}; "
            f"audio_backend={getattr(audio, '_backend', type(audio).__name__)}; "
            f"audio_fallback={type(getattr(audio, '_model', None)).__name__}; "
            f"text_backend={type(model.text_encoder.encoder).__name__}"
        )

    # --- Trainer ---
    from training.trainer import ConflictNetTrainer, get_warmup_cosine_scheduler
    from models.experiment_config import ExperimentConfig

    exp_config = ExperimentConfig.from_args(args)
    cfg = exp_config.to_dict()
    if getattr(args, "no_amp", False):
        cfg["amp"] = False
    else:
        cfg["amp"] = True
    if getattr(args, "compile", False):
        cfg["compile"] = True
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    trainer = ConflictNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        exp_config=exp_config,
        device=args.device,
        output_dir=args.output_dir,
    )

    start_epoch = 0
    if args.resume_from:
        start_epoch = trainer.load_checkpoint(args.resume_from)

    retries = 0

    while True:
        trainer.train(n_epochs=args.epochs or cfg.get("epochs", 30), pretrain_epochs=args.pretrain_epochs, start_epoch=start_epoch)

        if local_rank in (-1, 0):
            logger.info("⚡ Running final evaluation on best model...")
            # Try best_model first, fall back to latest_model if missing
            _best_ckpt = Path(args.output_dir) / "best_model.safetensors"
            _latest_ckpt = Path(args.output_dir) / "latest_model.safetensors"
            if _best_ckpt.exists():
                trainer.load_checkpoint(_best_ckpt)
            elif _latest_ckpt.exists():
                logger.warning("best_model.safetensors not found, falling back to latest_model.safetensors")
                trainer.load_checkpoint(_latest_ckpt)
            else:
                logger.warning("No checkpoint found for final evaluation, using current model state.")
            final_metrics = trainer.evaluate()

            print("\n📊 FINAL DETAILED EVALUATION RESULTS:")
            print("-" * 30)
            for k, v in final_metrics.items():
                print(f"{k:<15} : {v:.4f}")

        if args.max_retries <= 0 or args.target_f1 <= 0:
            break

        meta_path = Path(args.output_dir) / "best_model_meta.json"
        best_f1 = 0.0
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            best_f1 = meta.get("best_val_f1", 0.0)

        logger.info(f"[Retry {retries+1}/{args.max_retries}] Best val F1 = {best_f1:.4f}, target = {args.target_f1}")

        if best_f1 >= args.target_f1:
            logger.info(f"Target F1 {args.target_f1} reached!")
            break

        if retries >= args.max_retries:
            logger.info(f"Max retries ({args.max_retries}) exhausted, best F1 = {best_f1:.4f}")
            break

        retries += 1
        args.lr = float(args.lr) / 2
        args.resume_from = str(Path(args.output_dir) / "best_model.safetensors")
        args.pretrain_epochs = 0

        start_epoch = trainer.load_checkpoint(args.resume_from)
        args.epochs = start_epoch + args.resume_epochs

        logger.info(f"Resuming (retry {retries}/{args.max_retries}, lr={args.lr:.2e}, epochs {start_epoch}–{args.epochs-1})")

        # Reset scheduler for continuation — cosine decays new_lr → 0 over resume_epochs
        for g in trainer.optimizer.param_groups:
            g["lr"] = args.lr
        steps_per_epoch = len(trainer.train_loader)
        trainer.scheduler = get_warmup_cosine_scheduler(
            trainer.optimizer,
            num_warmup_steps=0,
            num_training_steps=steps_per_epoch * args.resume_epochs,
        )


def train_conflictnet(
    meld_root: Optional[str] = None,
    pt_dir: Optional[str] = None,
    output_dir: str = "checkpoints",
    batch_size: int = 16,
    epochs: int = 35,
    pretrain_epochs: int = 3,
    lr: float = 3e-5,
    audio_encoder: str = "wavlm_weighted",
    unfreeze_audio_layers: int = 0,
    amp: bool = True,
    compile: bool = False,
    iemocap_root: Optional[str] = None,
    mustard_root: Optional[str] = None,
    cremad_root: Optional[str] = None,
    **kwargs,
):
    """Python function interface to train ConflictNet without CLI flags.
    
    Usage:
        from scripts.train import train_conflictnet
        
        train_conflictnet(
            meld_root="D:/datasts/wav_dataset",
            pt_dir="D:/datasts/meld_features",
            batch_size=16,
            epochs=35,
            lr=2.5e-5,
        )
    """
    argv = []
    if meld_root:
        argv.extend(["--meld_root", str(meld_root)])
    if pt_dir:
        argv.extend(["--pt_dir", str(pt_dir)])
    if iemocap_root:
        argv.extend(["--iemocap_root", str(iemocap_root)])
    if mustard_root:
        argv.extend(["--mustard_root", str(mustard_root)])
    if cremad_root:
        argv.extend(["--cremad_root", str(cremad_root)])
    argv.extend(["--output_dir", str(output_dir)])
    argv.extend(["--batch_size", str(batch_size)])
    argv.extend(["--epochs", str(epochs)])
    argv.extend(["--pretrain_epochs", str(pretrain_epochs)])
    argv.extend(["--lr", str(lr)])
    argv.extend(["--audio_encoder", str(audio_encoder)])
    argv.extend(["--unfreeze_audio_layers", str(unfreeze_audio_layers)])
    if amp:
        argv.append("--amp")
    else:
        argv.append("--no_amp")
    if compile:
        argv.append("--compile")
    
    for k, v in kwargs.items():
        if isinstance(v, bool):
            if v:
                argv.append(f"--{k}")
        elif v is not None:
            argv.extend([f"--{k}", str(v)])
            
    args = parse_args(argv=argv)
    return main(args=args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # torchrun otherwise reports only a generic ChildFailedError after it
        # terminates the other workers, obscuring the rank-local root cause.
        logger.exception("Fatal training error on local rank %s", os.environ.get("LOCAL_RANK", "0"))
        raise
