"""Dataset loaders for ConflictNet training datasets.

Supported:
  - IEMOCAP: USC emotional speech, 10,039 utterances, 4/6 emotion classes
  - CREMA-D: 7,442 clips, 6 emotions, multiple speakers
  - MUStARD++: Sarcasm/non-sarcasm with audio + text + video (text+audio only used here)
  - MELD: Multimodal emotion in dialogue, 13,000+ utterances

Each dataset returns a dict with:
  - audio: (T_audio,) waveform tensor at 16kHz
  - input_ids: (seq_len,) tokenizer output
  - attention_mask: (seq_len,)
  - conflict_binary: int — 1 if conflict utterance
  - conflict_type_labels: (6,) multi-hot — [anger, disgust, fear, happiness, neutral, sadness]
    (based on CREMA-D's 6 emotion categories; non-CREMA-D datasets map to the nearest slot)
  - severity: float — [0, 1] intensity (if available)
  - speaker_id: str
  - gender: str or None

Collation: a custom collate_fn handles variable-length audio.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import re as _re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MAX_AUDIO_LEN = 10.0  # seconds
MAX_TEXT_LEN = 512

# Number of emotion categories (matches CREMA-D's 6 classes)
N_EMOTION_CLASSES = 6
# Label indices for each category (for use by all datasets)
EMOTION_IDX_ANGER = 0
EMOTION_IDX_DISGUST = 1
EMOTION_IDX_FEAR = 2
EMOTION_IDX_HAPPINESS = 3
EMOTION_IDX_NEUTRAL = 4
EMOTION_IDX_SADNESS = 5

IEMOCAP_EMOTION_MAP = {
    "ang": 0, "hap": 1, "exc": 1,  # merge excited→happy
    "sad": 2, "neu": 3, "fru": 4, "fea": 5, "sur": 6, "dis": 7, "oth": 8,
}
CONFLICT_EMOTIONS = {"ang", "fru"}  # used for binary conflict label in IEMOCAP


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def load_audio(path: str, target_sr: int = SAMPLE_RATE, max_len: float = MAX_AUDIO_LEN) -> torch.Tensor | Dict[str, torch.Tensor]:
    """Load and resample audio file to target_sr, or load precomputed .pt dict if exists."""
    pt_path = Path(path).with_suffix(".pt")
    
    pt_dir = os.environ.get("CONFLICTNET_PT_DIR", "/kaggle/working/features")
    if pt_dir and not pt_path.exists():
        pt_path = Path(pt_dir) / f"{Path(path).stem}.pt"

    if pt_path.exists():
        # Precomputed embedding dict, just load and return
        return torch.load(pt_path, map_location="cpu", weights_only=True)

    try:
        waveform, sr = torchaudio.load(path)
    except Exception:
        try:
            import soundfile as sf
            data, sr = sf.read(path)
            waveform = torch.from_numpy(data).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.t()
        except Exception as e:
            logger.warning(f"Failed to load audio from {path}: {e}, returning dummy audio tensor")
            waveform = torch.zeros(1, int(target_sr * max_len))
            sr = target_sr

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # stereo → mono
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    max_samples = int(max_len * target_sr)
    waveform = waveform[:, :max_samples]
    return waveform.squeeze(0)  # (T,)


def tokenize(
    text: str,
    tokenizer: AutoTokenizer,
    max_len: int = MAX_TEXT_LEN,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize text, return (input_ids, attention_mask)."""
    enc = tokenizer(  # type: ignore[operator]
        text,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)


# ---------------------------------------------------------------------------
# Word divergence helpers (MFA alignment pipeline)
# ---------------------------------------------------------------------------


def compute_token_word_boundaries(
    text: str,
    tokenizer: AutoTokenizer,
    max_len: int = MAX_TEXT_LEN,
) -> List[Tuple[int, int]]:
    """Map each word in text to its token span [token_start, token_end).

    Uses tokenizer ``return_offsets_mapping`` to align character-level
    word boundaries to token positions. Excludes special tokens (CLS, SEP).

    Args:
        text: The utterance text.
        tokenizer: HuggingFace tokenizer instance.
        max_len: Max token length (same as ``tokenize()``).

    Returns:
        List of ``(token_start_idx, token_end_idx)`` per word.
        Empty list if tokenization yields no content tokens.
    """
    enc = tokenizer(  # type: ignore[operator]
        text,
        max_length=max_len,
        truncation=True,
        padding=False,
        return_offsets_mapping=True,
    )
    offsets = enc.offset_mapping

    token_chars = []
    for i, (cs, ce) in enumerate(offsets):
        if cs is not None and ce is not None and cs < ce:
            token_chars.append((i, cs, ce))

    if not token_chars:
        return []

    word_spans = []
    for m in _re.finditer(r'\S+', text):
        ws, we = m.start(), m.end()
        if we > ws:
            word_spans.append((ws, we))

    boundaries = []
    for ws, we in word_spans:
        token_start = None
        token_end = None
        for idx, cs, ce in token_chars:
            if cs >= ws and cs < we:
                if token_start is None:
                    token_start = idx
                token_end = idx + 1
        if token_start is not None and token_end is not None and token_end > token_start:
            boundaries.append((token_start, token_end))

    return boundaries


def _load_word_timestamps_from_textgrid(
    textgrid_path: str,
) -> Optional[List[Tuple[float, float]]]:
    """Load per-word audio timestamps from an MFA ``.TextGrid`` file.

    Args:
        textgrid_path: Path to the ``.TextGrid`` file.

    Returns:
        List of ``(start_seconds, end_seconds)`` per word, or ``None``
        if the file is missing or unparseable.
    """
    if not os.path.isfile(textgrid_path):
        return None
    try:
        from models.alignment.word_divergence import parse_textgrid
        words = parse_textgrid(textgrid_path)
        if not words:
            return None
        return [(round(start, 3), round(end, 3)) for _, start, end in words]
    except Exception:
        logger.warning(f"Failed to parse TextGrid: {textgrid_path}")
        return None


def _textgrid_path_from_wav(
    wav_path: str,
    dataset_root: str,
    textgrid_root: Optional[str],
) -> Optional[str]:
    """Derive TextGrid path by mirroring wav directory structure.

    For ``wav_path`` rooted at ``dataset_root``, compute the relative
    path and look for a matching ``.TextGrid`` under ``textgrid_root``.
    """
    if textgrid_root is None or not wav_path:
        return None
    try:
        rel = os.path.relpath(wav_path, start=dataset_root)
    except ValueError:
        return os.path.join(textgrid_root, Path(wav_path).stem + ".TextGrid")
    tg_rel = os.path.splitext(rel)[0] + ".TextGrid"
    return os.path.join(textgrid_root, tg_rel)


# ---------------------------------------------------------------------------
# IEMOCAP
# ---------------------------------------------------------------------------

class IEMOCAPDataset(Dataset):
    """IEMOCAP dataset loader.

    Expects IEMOCAP data in the standard release directory structure:
      root/Session{1-5}/sentences/wav/{dialogue_id}/{utt_id}.wav
      root/Session{1-5}/dialog/EmoEvaluation/{dialogue_id}.txt
    """

    def __init__(
        self,
        root: str,
        tokenizer_name: str = "microsoft/deberta-v3-large",
        sessions: Optional[List[int]] = None,
        split: str = "train",
        conflict_as_anger_frustration: bool = True,
        textgrid_root: Optional[str] = None,
    ):
        self.root = Path(root)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.sessions = sessions or [1, 2, 3, 4, 5]
        self.conflict_as_anger_frustration = conflict_as_anger_frustration
        self.textgrid_root = textgrid_root
        self.items = self._scan_items()
        logger.info(f"[IEMOCAP] {split}: {len(self.items)} utterances")

    def _scan_items(self) -> List[Dict]:
        items = []
        for sess in self.sessions:
            sess_dir = self.root / f"Session{sess}"
            eval_dir = sess_dir / "dialog" / "EmoEvaluation"
            wav_root = sess_dir / "sentences" / "wav"
            transcript_root = sess_dir / "dialog" / "transcriptions"

            if not eval_dir.exists():
                logger.warning(f"[IEMOCAP] Missing evaluation dir: {eval_dir}")
                continue

            for eval_file in eval_dir.glob("*.txt"):
                dialogue_id = eval_file.stem
                conv_id = f"iemocap_{dialogue_id}"
                transcripts = self._load_transcriptions(
                    transcript_root / f"{dialogue_id}.txt"
                )
                turn_idx = 0
                with open(eval_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("%") or "\t" not in line:
                            continue
                        parts = line.split("\t")
                        if len(parts) < 4:
                            continue
                        utt_id = parts[1].strip()
                        emotion = parts[2].strip()
                        wav_path = wav_root / dialogue_id / f"{utt_id}.wav"
                        if not wav_path.exists():
                            continue
                        text = transcripts.get(utt_id, "")
                        if not self.conflict_as_anger_frustration:
                            raise NotImplementedError("IEMOCAP currently only supports conflict_as_anger_frustration=True")
                        conflict = emotion in CONFLICT_EMOTIONS
                        # IEMOCAP: anger→anger(0), frustration→anger(0), all others→neutral(4)
                        # Map IEMOCAP emotions to the 6-class CREMA-D emotion label space
                        type_labels = [0] * N_EMOTION_CLASSES
                        if emotion == "ang":
                            type_labels[EMOTION_IDX_ANGER] = 1
                        elif emotion == "fru":
                            type_labels[EMOTION_IDX_ANGER] = 1  # frustration → anger slot
                        elif emotion == "sad":
                            type_labels[EMOTION_IDX_SADNESS] = 1
                        elif emotion in ("hap", "exc"):
                            type_labels[EMOTION_IDX_HAPPINESS] = 1
                        elif emotion == "fea":
                            type_labels[EMOTION_IDX_FEAR] = 1
                        elif emotion == "dis":
                            type_labels[EMOTION_IDX_DISGUST] = 1
                        else:  # neu, sur, oth
                            type_labels[EMOTION_IDX_NEUTRAL] = 1
                        items.append({
                            "wav_path": str(wav_path),
                            "text": text,
                            "emotion": emotion,
                            "conflict_binary": int(conflict),
                            "conflict_type_labels": type_labels,
                            "severity": float(conflict),  # binary proxy; no real severity annotation in IEMOCAP
                        "speaker_id": utt_id[:6],  # e.g. 'Ses01F' — session+gender uniquely identifies speaker
                        "gender": "F" if utt_id[5] == "F" else "M",
                            "conversation_id": conv_id,
                            "turn_index": turn_idx,
                        })
                        turn_idx += 1
        return items

    def _load_transcriptions(self, path: Path) -> Dict[str, str]:
        transcripts = {}
        if not path.exists():
            return transcripts
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if " [" in line:
                    utt_id, rest = line.split(" [", 1)
                    text = rest.split("]:")[1].strip() if "]:" in rest else ""
                    transcripts[utt_id.strip()] = text
        return transcripts

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        audio = load_audio(item["wav_path"])
        input_ids, attention_mask = tokenize(item["text"], self.tokenizer)

        text = item["text"]
        word_timestamps: Optional[List[Tuple[float, float]]] = None
        token_word_boundaries: Optional[List[Tuple[int, int]]] = None
        if self.textgrid_root is not None:
            tg_path = _textgrid_path_from_wav(
                item["wav_path"], str(self.root), self.textgrid_root
            )
            if tg_path is not None:
                word_timestamps = _load_word_timestamps_from_textgrid(tg_path)
                if word_timestamps is not None and word_timestamps:
                    token_word_boundaries = compute_token_word_boundaries(text, self.tokenizer)

        return {
            "audio": audio,
            "audio_np": audio.numpy() if isinstance(audio, torch.Tensor) else None,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "conflict_binary": torch.tensor(item["conflict_binary"], dtype=torch.long),
            "conflict_type_labels": torch.tensor(item["conflict_type_labels"], dtype=torch.float),
            "severity": torch.tensor(item["severity"], dtype=torch.float),
            "speaker_id": item["speaker_id"],
            "gender": item["gender"],
            "text": text,
            "utterance_id": Path(item["wav_path"]).stem,
            "conversation_id": item.get("conversation_id", Path(item["wav_path"]).stem),
            "turn_index": item.get("turn_index", 0),
            "word_timestamps": word_timestamps,
            "token_word_boundaries": token_word_boundaries,
        }


# ---------------------------------------------------------------------------
# MUStARD++
# ---------------------------------------------------------------------------

class MUStARDDataset(Dataset):
    """MUStARD++ sarcasm dataset.

    Expects: root/mustard++_raw_data.json  (download from repo)
             root/{wav_dir}/{video_id}/{clip_id}.wav  (default wav_dir='utterances_final')
    """

    def __init__(
        self,
        root: str,
        json_file: str = "mustard_raw_data.json",
        tokenizer_name: str = "microsoft/deberta-v3-large",
        split: str = "train",
        train_ratio: float = 0.8,
        wav_dir: str = "utterances_final",
        wav_pattern: str = "*.wav",
        textgrid_root: Optional[str] = None,
    ):
        self.root = Path(root)
        self.wav_dir = Path(wav_dir) if not Path(wav_dir).is_absolute() else Path(wav_dir)
        self.wav_pattern = wav_pattern
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.textgrid_root = textgrid_root
        self.items = self._load_items(json_file, split, train_ratio)
        logger.info(f"[MUStARD++] {split}: {len(self.items)} utterances")

    def _load_items(self, json_file: str, split: str, train_ratio: float) -> List[Dict]:
        candidates = [
            self.root / json_file,
            self.root / "sarcasm_data.json",
            self.root / "data" / "sarcasm_data.json",
            self.root / "data" / "mustard_raw_data.json"
        ]
        json_path = None
        for c in candidates:
            if c.exists():
                json_path = c
                break
                
        if json_path is None:
            logger.warning(f"[MUStARD++] JSON not found in {self.root}")
            return []
            
        with open(json_path) as f:
            data = json.load(f)

        wav_search_root = self.root / self.wav_dir if not self.wav_dir.is_absolute() else self.wav_dir
        if not wav_search_root.exists():
            wav_search_root = self.root
        logger.info(f"[MUStARD++] Searching for wavs in {wav_search_root} with pattern {self.wav_pattern}")

        # Index all audio/video files once to avoid thousand+ disk traversals on Kaggle read-only mounts
        wav_index: Dict[str, Path] = {}
        valid_exts = {".wav", ".mp4", ".mkv", ".avi", ".mp3", ".flac", ".m4a", ".aac"}
        
        if wav_search_root.exists():
            logger.info(f"[MUStARD++] Indexing media files in {wav_search_root}...")
            try:
                import os
                for f in os.listdir(str(wav_search_root)):
                    p = Path(wav_search_root) / f
                    if p.is_file() and p.suffix.lower() in valid_exts:
                        wav_index[p.stem] = p
                        wav_index[p.name] = p
                        if p.stem.endswith("_u"):
                            wav_index[p.stem[:-2]] = p
            except Exception as e:
                logger.warning(f"[MUStARD++] Error scanning {wav_search_root}: {e}")
            if wav_index:
                logger.info(f"[MUStARD++] Indexed {len(wav_index)} media files from {wav_search_root} (non-recursive)")

        if not wav_index:
            logger.warning(f"[MUStARD++] WARNING: 0 media files found in {wav_search_root}! Dataset will be empty.")

        all_samples = []
        for key, sample in data.items():
            wav_path = wav_index.get(key)
            if wav_path is None:
                # Key without '_u' or with '_u' suffix handling
                alt_key = key[:-2] if key.endswith("_u") else f"{key}_u"
                wav_path = wav_index.get(alt_key)
            if wav_path is None:
                # Fuzzy fallback matching on stems
                for stem, p in wav_index.items():
                    if key in stem or stem in key:
                        wav_path = p
                        break
            if wav_path is None:
                logger.warning(f"[MUStARD++] No wav found for key={key}, searched in {wav_search_root}")
                continue
            all_samples.append({
                "wav_path": str(wav_path),
                "text": sample.get("utterance", ""),
                "sarcasm": int(sample.get("sarcasm", 0)),
                "speaker_id": sample.get("speaker", "unknown"),
                "dataset_name": "mustard",
            })

        # Speaker-stratified split: group by speaker, assign whole speakers to train/val
        speaker_groups: Dict[str, List[Dict]] = {}
        for s in all_samples:
            speaker_groups.setdefault(s["speaker_id"], []).append(s)
        rng = random.Random(42)  # deterministic shuffle for reproducibility
        speaker_ids = list(speaker_groups.keys())
        rng.shuffle(speaker_ids)

        train_items: List[Dict] = []
        val_items: List[Dict] = []
        target_train = int(len(all_samples) * train_ratio)
        for sid in speaker_ids:
            samples = speaker_groups[sid]
            if not train_items or len(train_items) + len(samples) < target_train:
                train_items.extend(samples)
            else:
                val_items.extend(samples)
        return train_items if split == "train" else val_items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        audio = load_audio(item["wav_path"])
        input_ids, attention_mask = tokenize(item["text"], self.tokenizer)
        sarcasm = item["sarcasm"]
        text = item["text"]
        word_timestamps: Optional[List[Tuple[float, float]]] = None
        token_word_boundaries: Optional[List[Tuple[int, int]]] = None
        if self.textgrid_root is not None:
            tg_path = _textgrid_path_from_wav(
                item["wav_path"], str(self.root), self.textgrid_root
            )
            if tg_path is not None:
                word_timestamps = _load_word_timestamps_from_textgrid(tg_path)
                if word_timestamps is not None and word_timestamps:
                    token_word_boundaries = compute_token_word_boundaries(text, self.tokenizer)
        return {
            "audio": audio,
            "audio_np": audio.numpy() if isinstance(audio, torch.Tensor) else None,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "conflict_binary": torch.tensor(sarcasm, dtype=torch.long),
            # MUStARD++: sarcasm maps to anger slot (0) as closest conflict proxy; no direct CREMA-D mapping
            "conflict_type_labels": torch.tensor([sarcasm, 0, 0, 0, 0, 0], dtype=torch.float),
            "severity": torch.tensor(float(sarcasm), dtype=torch.float),  # binary proxy; no real severity in MUStARD++
            "speaker_id": item["speaker_id"],
            "gender": None,
            "text": item["text"],
            "utterance_id": Path(item["wav_path"]).stem,
            "conversation_id": item.get("conversation_id", Path(item["wav_path"]).stem),
            "turn_index": item.get("turn_index", 0),
            "word_timestamps": word_timestamps,
            "token_word_boundaries": token_word_boundaries,
            "dataset_name": item.get("dataset_name", "unknown"),
        }


# ---------------------------------------------------------------------------
# CREMA-D
# ---------------------------------------------------------------------------

# CREMA-D 6 emotion categories — each maps to its index in conflict_type_labels
# Index order: [anger=0, disgust=1, fear=2, happiness=3, neutral=4, sadness=5]
CREMAD_EMOTIONS = {
    "ANG": "anger", "DIS": "disgust", "FEA": "fear",
    "HAP": "happiness", "NEU": "neutral", "SAD": "sadness",
}
CREMAD_EMOTION_IDX = {
    "ANG": 0, "DIS": 1, "FEA": 2,
    "HAP": 3, "NEU": 4, "SAD": 5,
}
CREMAD_CONFLICT_EMOTIONS = {"ANG", "DIS", "FEA"}  # anger, disgust, fear → binary conflict


class CREMADDataset(Dataset):
    """CREMA-D dataset loader.

    Expects: root/AudioWAV/{actor}_{sentence}_{emotion}_{level}.wav
    Download from: https://github.com/CheyneyComputerScience/CREMA-D
    """

    def __init__(
        self,
        root: str,
        tokenizer_name: str = "microsoft/deberta-v3-large",
        split: str = "train",
        train_ratio: float = 0.8,
        textgrid_root: Optional[str] = None,
    ):
        self.root = Path(root)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.textgrid_root = textgrid_root
        self.items = self._scan_items(split, train_ratio)
        logger.info(f"[CREMA-D] {split}: {len(self.items)} utterances")

    # CREMA-D sentences (12 fixed sentences used in recordings)
    _SENTENCES = {
        "IEO": "It's eleven o'clock.",
        "TIE": "That is exactly what happened.",
        "IOM": "I'm on my way to the meeting.",
        "IWW": "I wonder what this is about.",
        "TAI": "The airplane is almost full.",
        "MTI": "Maybe tomorrow it will be cold.",
        "IWL": "I would like a new alarm clock.",
        "ITH": "I think I have a doctor's appointment.",
        "DFA": "Don't forget a jacket.",
        "ITS": "I think I've seen this before.",
        "TSI": "The surface is slippery.",
        "WSI": "We'll stop in a couple of minutes.",
    }

    def _scan_items(self, split: str, train_ratio: float) -> List[Dict]:
        wav_dir = self.root / "AudioWAV"
        if not wav_dir.exists():
            logger.warning(f"[CREMA-D] Missing AudioWAV dir: {wav_dir}")
            return []

        all_wavs = list(wav_dir.glob("*.wav"))

        # Collect all samples with actor info
        all_samples = []
        for wav in all_wavs:
            parts = wav.stem.split("_")
            if len(parts) < 4:
                continue
            actor_id = parts[0]
            sentence_id = parts[1]
            emotion = parts[2]
            text = self._SENTENCES.get(sentence_id, "")
            conflict = emotion in CREMAD_CONFLICT_EMOTIONS
            # CREMA-D: emit a proper 6-class one-hot label using the emotion index
            type_labels = [0] * N_EMOTION_CLASSES
            emo_idx = CREMAD_EMOTION_IDX.get(emotion)
            if emo_idx is not None:
                type_labels[emo_idx] = 1
            all_samples.append({
                "wav_path": str(wav),
                "text": text,
                "emotion": CREMAD_EMOTIONS.get(emotion, emotion.lower()),
                "conflict_binary": int(conflict),
                "conflict_type_labels": type_labels,  # 6-class one-hot: [anger, disgust, fear, happiness, neutral, sadness]
                "severity": float(conflict),
                "speaker_id": f"cremad_{actor_id}",
                "gender": None,
            })

        # Speaker-stratified split: group by actor, assign whole actors to train/val
        speaker_groups: Dict[str, List[Dict]] = {}
        for s in all_samples:
            speaker_groups.setdefault(s["speaker_id"], []).append(s)
        rng = random.Random(42)
        speaker_ids = list(speaker_groups.keys())
        rng.shuffle(speaker_ids)

        train_items: List[Dict] = []
        val_items: List[Dict] = []
        target_train = int(len(all_samples) * train_ratio)
        for sid in speaker_ids:
            samples = speaker_groups[sid]
            if not train_items or len(train_items) + len(samples) < target_train:
                train_items.extend(samples)
            else:
                val_items.extend(samples)
        return train_items if split == "train" else val_items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        audio = load_audio(item["wav_path"])
        input_ids, attention_mask = tokenize(item["text"], self.tokenizer)
        text = item["text"]
        word_timestamps: Optional[List[Tuple[float, float]]] = None
        token_word_boundaries: Optional[List[Tuple[int, int]]] = None
        if self.textgrid_root is not None:
            tg_path = _textgrid_path_from_wav(
                item["wav_path"], str(self.root), self.textgrid_root
            )
            if tg_path is not None:
                word_timestamps = _load_word_timestamps_from_textgrid(tg_path)
                if word_timestamps is not None and word_timestamps:
                    token_word_boundaries = compute_token_word_boundaries(text, self.tokenizer)
        return {
            "audio": audio,
            "audio_np": audio.numpy() if isinstance(audio, torch.Tensor) else None,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "conflict_binary": torch.tensor(item["conflict_binary"], dtype=torch.long),
            "conflict_type_labels": torch.tensor(item["conflict_type_labels"], dtype=torch.float),
            "severity": torch.tensor(item["severity"], dtype=torch.float),
            "speaker_id": item["speaker_id"],
            "gender": item["gender"],
            "text": item["text"],
            "utterance_id": Path(item["wav_path"]).stem,
            "conversation_id": Path(item["wav_path"]).stem,
            "turn_index": 0,
            "word_timestamps": word_timestamps,
            "token_word_boundaries": token_word_boundaries,
        }


# ---------------------------------------------------------------------------
# MELD (Multimodal EmotionLines Dataset)
# ---------------------------------------------------------------------------

MELD_CONFLICT_EMOTIONS = {"anger", "disgust", "fear"}


class MELDDataset(Dataset):
    """MELD dataset loader with dialogue context.

    Expects standard MELD directory structure:
      root/train/train_sent_emo.csv  (or dev/test)
      root/train/train_splits/dia{N}_utt{M}.wav

    Download from: https://affective-meld.github.io/
    """

    def __init__(
        self,
        root: str,
        tokenizer_name: str = "microsoft/deberta-v3-large",
        split: str = "train",
        textgrid_root: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        self.root = Path(root)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.split = split
        self.textgrid_root = textgrid_root
        self.max_samples = max_samples
        self.items = self._load_items()
        logger.info(f"[MELD] {split}: {len(self.items)} utterances")

    def _load_items(self) -> List[Dict]:
        split_map = {"train": "train", "val": "dev", "test": "test"}
        split_name = split_map.get(self.split, self.split)
        
        # Look for CSV in standard locations
        csv_candidates = [
            self.root / split_name / f"{split_name}_sent_emo.csv",
            self.root / f"{split_name}_sent_emo.csv",
            self.root / f"{self.split}_sent_emo.csv",
        ]
        csv_path = None
        for c in csv_candidates:
            if c.exists():
                csv_path = c
                break

        if csv_path is None or not csv_path.exists():
            logger.warning(f"[MELD] CSV not found in {self.root} for split {split_name}")
            return []

        # Hardcode split folder paths based on Kaggle MELD structure
        if split_name == "train":
            split_folder = self.root / "train" / "train_splits"
        elif split_name == "dev":
            split_folder = self.root / "dev" / "dev_splits_complete"
        elif split_name == "test":
            # Kaggle UI might truncate the name; check likely candidates
            split_folder = self.root / "test" / "output_repeated_splits_test"
            if not split_folder.exists():
                split_folder = self.root / "test" / "output_repeated_splits"
            if not split_folder.exists():
                split_folder = self.root / "test" / "output_repeated_spl"
        else:
            split_folder = self.root / split_name

        items = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            available = [c.strip() for c in reader.fieldnames] if reader.fieldnames else []

            def _find_field(candidates):
                for candidate in candidates:
                    for fname in available:
                        if fname.lower().replace("-", "_").replace(" ", "_") == candidate.lower().replace("-", "_").replace(" ", "_"):
                            return fname
                return None

            field_map = {}
            required = {
                "dialogue_id": ["Dialogue_ID", "dialogue_id", "DialogueId"],
                "utterance_id": ["Utterance_ID", "utterance_id", "UtteranceId"],
                "utterance": ["Utterance", "utterance", "text", "Text"],
                "emotion": ["Emotion", "emotion", "Sentiment", "sentiment"],
                "speaker": ["Speaker", "speaker", "Speaker_ID", "speaker_id"],
            }
            for key, candidates in required.items():
                found = _find_field(candidates)
                if found is None:
                    raise KeyError(
                        f"[MELD] Required CSV field not found in {csv_path}. "
                        f"Looked for {candidates}, found columns: {available}"
                    )
                field_map[key] = found

            for row in reader:
                dia_id = str(row.get(field_map["dialogue_id"], "")).strip()
                utt_id = str(row.get(field_map["utterance_id"], "")).strip()
                text = str(row.get(field_map["utterance"], "")).strip()
                emotion = str(row.get(field_map["emotion"], "neutral")).strip().lower()
                speaker = str(row.get(field_map["speaker"], "unknown")).strip()

                wav_path = split_folder / f"dia{dia_id}_utt{utt_id}.mp4"
                if not wav_path.exists():
                    continue

                conflict = emotion in MELD_CONFLICT_EMOTIONS
                # MELD: map emotion to 6-class CREMA-D label space
                type_labels = [0] * N_EMOTION_CLASSES
                if emotion == "anger":
                    type_labels[EMOTION_IDX_ANGER] = 1
                elif emotion == "disgust":
                    type_labels[EMOTION_IDX_DISGUST] = 1
                elif emotion == "fear":
                    type_labels[EMOTION_IDX_FEAR] = 1
                elif emotion in ("joy", "surprise"):
                    type_labels[EMOTION_IDX_HAPPINESS] = 1
                elif emotion == "sadness":
                    type_labels[EMOTION_IDX_SADNESS] = 1
                else:  # neutral
                    type_labels[EMOTION_IDX_NEUTRAL] = 1
                items.append({
                    "wav_path": str(wav_path),
                    "text": text,
                    "emotion": emotion,
                    "dialogue_id": dia_id,
                    "utterance_id": utt_id,
                    "turn_index": int(utt_id) if utt_id.isdigit() else 0,
                    "conflict_binary": int(conflict),
                    "conflict_type_labels": type_labels,  # 6-class one-hot: [anger, disgust, fear, happiness, neutral, sadness]
                    "severity": float(conflict),  # binary proxy; no real severity in MELD
                    "speaker_id": f"meld_{speaker}",
                    "gender": None,
                    "dataset_name": "meld",
                })
        if self.max_samples is not None and len(items) > self.max_samples:
            total = len(items)
            # Group items by dialogue_id to preserve chronological context
            dialogues = {}
            for item in items:
                d_id = item["dialogue_id"]
                if d_id not in dialogues:
                    dialogues[d_id] = []
                dialogues[d_id].append(item)

            # Deterministic shuffle of dialogue IDs
            dialogue_ids = sorted(list(dialogues.keys()))
            rng = random.Random(42)
            rng.shuffle(dialogue_ids)

            subsampled_items = []
            n_conflict = 0
            n_non_conflict = 0

            for d_id in dialogue_ids:
                if len(subsampled_items) >= self.max_samples:
                    break
                # Ensure turns within the dialogue are in chronological order
                d_items = sorted(dialogues[d_id], key=lambda x: x["turn_index"])
                subsampled_items.extend(d_items)
                
                for x in d_items:
                    if x["conflict_binary"] == 1:
                        n_conflict += 1
                    else:
                        n_non_conflict += 1

            items = subsampled_items
            logger.info(
                f"[MELD] {self.split}: subsampled to {len(items)} "
                f"({n_conflict} conflict, {n_non_conflict} non-conflict) "
                f"from {total} total (max_samples={self.max_samples}) "
                f"preserving {len(set(x['dialogue_id'] for x in items))} full dialogues."
            )
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        audio = load_audio(item["wav_path"])
        input_ids, attention_mask = tokenize(item["text"], self.tokenizer)
        text = item["text"]
        word_timestamps: Optional[List[Tuple[float, float]]] = None
        token_word_boundaries: Optional[List[Tuple[int, int]]] = None
        if self.textgrid_root is not None:
            tg_path = _textgrid_path_from_wav(
                item["wav_path"], str(self.root), self.textgrid_root
            )
            if tg_path is not None:
                word_timestamps = _load_word_timestamps_from_textgrid(tg_path)
                if word_timestamps is not None and word_timestamps:
                    token_word_boundaries = compute_token_word_boundaries(text, self.tokenizer)
        return {
            "audio": audio,
            "audio_np": audio.numpy() if isinstance(audio, torch.Tensor) else None,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "conflict_binary": torch.tensor(item["conflict_binary"], dtype=torch.long),
            "conflict_type_labels": torch.tensor(item["conflict_type_labels"], dtype=torch.float),
            "severity": torch.tensor(item["severity"], dtype=torch.float),
            "speaker_id": item["speaker_id"],
            "gender": item["gender"],
            "text": item["text"],
            "utterance_id": Path(item["wav_path"]).stem,
            "conversation_id": f"meld_{item['dialogue_id']}",
            "turn_index": item.get("turn_index", 0),
            "word_timestamps": word_timestamps,
            "token_word_boundaries": token_word_boundaries,
            "dataset_name": item.get("dataset_name", "unknown"),
        }


# ---------------------------------------------------------------------------
# CMU-MOSEI
# ---------------------------------------------------------------------------

CMUMOSEI_CONFLICT_EMOTIONS = {"anger", "disgust", "fear"}


class CMUMOSEIDataset(Dataset):
    """CMU-MOSEI dataset loader.

    Expects the standard CMU Multimodal SDK format:
      root/CMU_MOSEI/Labeled/{split}.csv
      root/CMU_MOSEI/Audio/WAV_16000/{utterance_id}.wav

    Each CSV row: utterance_id, text, emotion (lowercase label).

    Download from: https://github.com/A2Zadeh/CMU-MultimodalSDK
    """

    def __init__(
        self,
        root: str,
        tokenizer_name: str = "microsoft/deberta-v3-large",
        split: str = "train",
        textgrid_root: Optional[str] = None,
    ):
        self.root = Path(root)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.split = split
        self.textgrid_root = textgrid_root
        self.items = self._load_items()
        logger.info(f"[CMU-MOSEI] {split}: {len(self.items)} utterances")

    def _load_items(self) -> List[Dict]:
        csv_path = self.root / "CMU_MOSEI" / "Labeled" / f"{self.split}.csv"
        wav_root = self.root / "CMU_MOSEI" / "Audio" / "WAV_16000"

        if not csv_path.exists():
            logger.warning(f"[CMU-MOSEI] CSV not found: {csv_path}")
            return []

        items = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            available = [c.strip() for c in reader.fieldnames] if reader.fieldnames else []

            def _find_field(candidates):
                for candidate in candidates:
                    for fname in available:
                        if fname.lower().replace("-", "_").replace(" ", "_") == candidate.lower().replace("-", "_").replace(" ", "_"):
                            return fname
                return None

            required = {
                "utterance_id": ["utterance_id", "Utterance_ID", "id", "ID"],
                "text": ["text", "Text", "utterance", "Utterance"],
                "emotion": ["emotion", "Emotion", "label", "Label"],
            }
            field_map = {}
            for key, candidates in required.items():
                found = _find_field(candidates)
                if found is None:
                    raise KeyError(
                        f"[CMU-MOSEI] Required CSV field not found in {csv_path}. "
                        f"Looked for {candidates}, found columns: {available}"
                    )
                field_map[key] = found

            for row in reader:
                uid = row.get(field_map["utterance_id"], "").strip()
                text = row.get(field_map["text"], "").strip()
                emotion = row.get(field_map["emotion"], "neutral").strip().lower()

                if not uid or not text:
                    continue

                wav_path = wav_root / f"{uid}.wav"
                try:
                    if not wav_path.exists():
                        logger.debug(f"[CMU-MOSEI] Missing wav: {wav_path}")
                        continue
                except OSError:
                    continue

                conflict = emotion in CMUMOSEI_CONFLICT_EMOTIONS
                # CMU-MOSEI utterance IDs: {video_id}_{segment}; use video_id as speaker proxy
                speaker_prefix = uid.split("_")[0] if "_" in uid else uid
                # Map CMU-MOSEI emotion to 6-class CREMA-D label space
                type_labels = [0] * N_EMOTION_CLASSES
                if emotion == "anger":
                    type_labels[EMOTION_IDX_ANGER] = 1
                elif emotion == "disgust":
                    type_labels[EMOTION_IDX_DISGUST] = 1
                elif emotion == "fear":
                    type_labels[EMOTION_IDX_FEAR] = 1
                elif emotion in ("happiness", "happy"):
                    type_labels[EMOTION_IDX_HAPPINESS] = 1
                elif emotion in ("sadness", "sad"):
                    type_labels[EMOTION_IDX_SADNESS] = 1
                else:  # neutral and others
                    type_labels[EMOTION_IDX_NEUTRAL] = 1
                items.append({
                    "wav_path": str(wav_path),
                    "text": text,
                    "emotion": emotion,
                    "utterance_id": uid,
                    "conflict_binary": int(conflict),
                    "conflict_type_labels": type_labels,  # 6-class: [anger, disgust, fear, happiness, neutral, sadness]
                    "severity": float(conflict),  # binary proxy; no real severity in CMU-MOSEI
                    "speaker_id": f"mosei_{speaker_prefix}",
                    "gender": None,
                })
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        try:
            audio = load_audio(item["wav_path"])
        except Exception as e:
            logger.warning(f"[CMU-MOSEI] Failed to load audio at idx {idx}: {e}")
            audio = torch.zeros(int(SAMPLE_RATE * 1.0))
        input_ids, attention_mask = tokenize(item["text"], self.tokenizer)
        text = item["text"]
        word_timestamps: Optional[List[Tuple[float, float]]] = None
        token_word_boundaries: Optional[List[Tuple[int, int]]] = None
        if self.textgrid_root is not None:
            tg_path = _textgrid_path_from_wav(
                item["wav_path"], str(self.root), self.textgrid_root
            )
            if tg_path is not None:
                word_timestamps = _load_word_timestamps_from_textgrid(tg_path)
                if word_timestamps is not None and word_timestamps:
                    token_word_boundaries = compute_token_word_boundaries(text, self.tokenizer)
        return {
            "audio": audio,
            "audio_np": audio.numpy() if isinstance(audio, torch.Tensor) else None,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "conflict_binary": torch.tensor(item["conflict_binary"], dtype=torch.long),
            "conflict_type_labels": torch.tensor(item["conflict_type_labels"], dtype=torch.float),
            "severity": torch.tensor(item["severity"], dtype=torch.float),
            "speaker_id": item["speaker_id"],
            "gender": item["gender"],
            "text": item["text"],
            "utterance_id": Path(item["wav_path"]).stem,
            "conversation_id": f"mosei_{Path(item['wav_path']).stem.split('_')[0]}",
            "turn_index": int(Path(item['wav_path']).stem.split('_')[1]) if '_' in Path(item['wav_path']).stem and Path(item['wav_path']).stem.split('_')[1].isdigit() else 0,
            "word_timestamps": word_timestamps,
            "token_word_boundaries": token_word_boundaries,
        }


# ---------------------------------------------------------------------------
# CASE 2026 — Conflict-Aware Sarcasm Evaluation Benchmark
# ---------------------------------------------------------------------------

CASE_CONFLICT_EMOTIONS = {"sarcasm", "sarcastic", "angry", "frustrated", "disgust"}
# CASE → 6-class CREMA-D emotion label space: [anger, disgust, fear, happiness, neutral, sadness]
CASE_TYPE_MAP = {
    "sarcasm":    [1, 0, 0, 0, 0, 0],  # anger slot (closest conflict proxy)
    "sarcastic":  [1, 0, 0, 0, 0, 0],
    "angry":      [1, 0, 0, 0, 0, 0],  # anger
    "frustrated": [1, 0, 0, 0, 0, 0],  # anger (closest)
    "disgust":    [0, 1, 0, 0, 0, 0],  # disgust
    "fear":       [0, 0, 1, 0, 0, 0],  # fear
    "happy":      [0, 0, 0, 1, 0, 0],  # happiness
    "surprise":   [0, 0, 0, 1, 0, 0],  # happiness (closest)
    "neutral":    [0, 0, 0, 0, 1, 0],  # neutral
    "sad":        [0, 0, 0, 0, 0, 1],  # sadness
}


class CASEDataset(Dataset):
    """CASE 2026 benchmark — Conflict-Aware Sarcasm Evaluation.

    Expected directory structure::

        {root}/
            metadata.jsonl        — one JSON object per line
            wav/                  — 16kHz mono WAV files referenced by utterance_id

    Each JSONL line has the following fields:

    .. code-block:: json

        {
            "utterance_id": "case_00001",
            "text": "Great, another meeting...",
            "emotion": "sarcasm",
            "speaker_id": "speaker_042",
            "gender": "F",
            "severity": 0.85
        }

    Subtype mapping to 6-class CREMA-D label space [anger, disgust, fear, happiness, neutral, sadness]:
        - sarcasm / sarcastic / angry / frustrated → [1, 0, 0, 0, 0, 0] (anger slot)
        - disgust → [0, 1, 0, 0, 0, 0]
        - fear → [0, 0, 1, 0, 0, 0]
        - happy / surprise → [0, 0, 0, 1, 0, 0] (happiness slot)
        - neutral → [0, 0, 0, 0, 1, 0]
        - sad → [0, 0, 0, 0, 0, 1]

    Download: https://github.com/anonymous/case2026 (placeholder)
    """

    def __init__(
        self,
        root: str,
        tokenizer_name: str = "microsoft/deberta-v3-large",
        split: str = "train",
        max_samples: Optional[int] = None,
        textgrid_root: Optional[str] = None,
    ):
        self.root = Path(root)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.split = split
        self.textgrid_root = textgrid_root
        self.items = self._load_items(max_samples)
        logger.info(f"[CASE] {split}: {len(self.items)} utterances")

    def _load_items(self, max_samples: Optional[int]) -> List[Dict]:
        meta_path = self.root / "metadata.jsonl"
        if not meta_path.exists():
            logger.warning(f"[CASE] metadata not found: {meta_path}")
            return []

        wav_root = self.root / "wav"
        if not wav_root.exists():
            logger.warning(f"[CASE] wav directory not found: {wav_root}")
            return []

        items = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples is not None and len(items) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                uid = row.get("utterance_id", f"case_{i:06d}")
                text = row.get("text", "").strip()
                emotion = row.get("emotion", "neutral").strip().lower()
                speaker_id = row.get("speaker_id", f"unknown_{i}")
                gender = row.get("gender", None)
                severity = float(row.get("severity", 0.7 if emotion in CASE_CONFLICT_EMOTIONS else 0.1))
                type_labels = CASE_TYPE_MAP.get(emotion, [0, 0, 0, 0, 1, 0])  # default: neutral slot
                conflict_binary = 1 if emotion in CASE_CONFLICT_EMOTIONS else 0

                wav_path = wav_root / f"{uid}.wav"
                if not wav_path.exists():
                    logger.debug(f"[CASE] Missing wav: {wav_path}")
                    wav_path = None

                items.append({
                    "wav_path": str(wav_path) if wav_path else None,
                    "text": text,
                    "emotion": emotion,
                    "utterance_id": uid,
                    "conflict_binary": conflict_binary,
                    "conflict_type_labels": type_labels,
                    "severity": severity,
                    "speaker_id": speaker_id,
                    "gender": gender,
                })
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        if item["wav_path"] is not None:
            try:
                audio = load_audio(item["wav_path"])
            except Exception as e:
                logger.warning(f"[CASE] Failed to load audio idx {idx}: {e}")
                audio = torch.zeros(int(SAMPLE_RATE * 1.0))
        else:
            audio = torch.zeros(int(SAMPLE_RATE * 1.0))
        input_ids, attention_mask = tokenize(item["text"], self.tokenizer)
        text = item["text"]
        word_timestamps: Optional[List[Tuple[float, float]]] = None
        token_word_boundaries: Optional[List[Tuple[int, int]]] = None
        if self.textgrid_root is not None and item["wav_path"] is not None:
            tg_path = _textgrid_path_from_wav(
                item["wav_path"], str(self.root), self.textgrid_root
            )
            if tg_path is not None:
                word_timestamps = _load_word_timestamps_from_textgrid(tg_path)
                if word_timestamps is not None and word_timestamps:
                    token_word_boundaries = compute_token_word_boundaries(text, self.tokenizer)
        return {
            "audio": audio,
            "audio_np": audio.numpy() if isinstance(audio, torch.Tensor) else None,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "conflict_binary": torch.tensor(item["conflict_binary"], dtype=torch.long),
            "conflict_type_labels": torch.tensor(item["conflict_type_labels"], dtype=torch.float),
            "severity": torch.tensor(item["severity"], dtype=torch.float),
            "speaker_id": item["speaker_id"],
            "gender": item["gender"],
            "text": item["text"],
            "utterance_id": Path(item["wav_path"]).stem if item["wav_path"] is not None else f"case_{idx}",
            "conversation_id": Path(item["wav_path"]).stem if item["wav_path"] is not None else f"case_{idx}",
            "turn_index": 0,
            "word_timestamps": word_timestamps,
            "token_word_boundaries": token_word_boundaries,
        }


# ---------------------------------------------------------------------------
# GoEmotions (text-only, for encoder pre-training)
# ---------------------------------------------------------------------------


class GoEmotionsDataset(Dataset):
    """GoEmotions text-only dataset for encoder pre-training.

    No audio — returns dummy zero audio. Used only for text encoder
    pre-training to augment swap-objective data. All samples are
    non-conflict (conflict_binary=0).

    Uses: https://huggingface.co/datasets/go_emotions
    """

    def __init__(
        self,
        tokenizer_name: str = "microsoft/deberta-v3-large",
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.split = split
        self.items = self._load_items(max_samples)
        logger.info(f"[GoEmotions] {split}: {len(self.items)} utterances")

    def _load_items(self, max_samples: Optional[int]) -> List[Dict]:
        try:
            from datasets import load_dataset
            ds = load_dataset("go_emotions", split=self.split)
        except Exception as e:
            logger.warning(f"[GoEmotions] Failed to load dataset: {e}")
            return []

        items = []
        for i, example in enumerate(ds):
            if max_samples is not None and i >= max_samples:
                break
            text = example.get("text", "").strip()  # type: ignore[attr-defined]
            if not text:
                continue
            items.append({
                "text": text,
            })
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        dummy_audio = torch.zeros(SAMPLE_RATE)  # 1 second of silence
        input_ids, attention_mask = tokenize(item["text"], self.tokenizer)
        return {
            "audio": dummy_audio,
            "audio_np": np.zeros(SAMPLE_RATE, dtype=np.float32),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "conflict_binary": torch.tensor(0, dtype=torch.long),
            "conflict_type_labels": torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float),
            "severity": torch.tensor(0.0, dtype=torch.float),
            "speaker_id": f"goemotions_{idx}",
            "gender": None,
            "text": item["text"],
            "utterance_id": f"goemotions_{idx}",
            # Each GoEmotions sample is treated as its own single-turn conversation
            # so the context cache never accumulates cross-sample history.
            "conversation_id": f"goemotions_{idx}",
            "turn_index": 0,
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def _collate_core(
    batch: List[Dict[str, Any]],
    augmentor: Any = None,
    prosody_lookup: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Pad variable-length audio; optionally augment; look up prosody z-scores.

    Args:
        batch: List of per-sample dicts from the dataset.
        augmentor: An ``AudioAugmentor`` instance (or None to skip augmentation).
        prosody_lookup: Pre-computed prosody z-scores dict mapping
            ``{utterance_id: (3,) tensor}``, keyed by audio file stem.
            Generated offline by ``scripts/compute_prosody_stats.py``.
            If None, z-scores default to zeros.
    """
    batch_out = []
    if augmentor is not None and augmentor.available:
        for b in batch:
            if b.get("audio_np") is not None:
                aug_np = augmentor(b["audio_np"])
                b_new = {**b, "audio_np": aug_np, "audio": torch.tensor(aug_np, dtype=torch.float32)}
                batch_out.append(b_new)
            else:
                batch_out.append(b)
        batch = batch_out

    # Check if we loaded precomputed dicts instead of raw audio tensors
    # Safely handle mixed batches by defaulting missing .pt files to zero tensors
    is_precomputed = isinstance(batch[0]["audio"], dict)
    
    if is_precomputed:
        audio_padded = torch.stack([
            b["audio"]["audio"] if isinstance(b["audio"], dict) else torch.zeros(256) 
            for b in batch
        ])
        speaker_padded = torch.stack([
            b["audio"]["speaker"] if isinstance(b["audio"], dict) else torch.zeros(192) 
            for b in batch
        ])
        
        has_frames = any("audio_frames" in b["audio"] and b["audio"]["audio_frames"] is not None for b in batch if isinstance(b["audio"], dict))
        if has_frames:
            max_f = max(b["audio"]["audio_frames"].shape[0] for b in batch if isinstance(b["audio"], dict) and b["audio"].get("audio_frames") is not None)
            D = next(b["audio"]["audio_frames"].shape[1] for b in batch if isinstance(b["audio"], dict) and b["audio"].get("audio_frames") is not None)
            audio_frames_padded = torch.zeros(len(batch), max_f, D)
            audio_attention_mask = torch.zeros(len(batch), max_f, dtype=torch.bool)
            for i, b in enumerate(batch):
                if isinstance(b["audio"], dict) and b["audio"].get("audio_frames") is not None:
                    f = b["audio"]["audio_frames"]
                    t = f.shape[0]
                    audio_frames_padded[i, :t] = f
                    audio_attention_mask[i, :t] = True
        else:
            audio_frames_padded = None
            audio_attention_mask = torch.ones(len(batch), 1, dtype=torch.bool)
    else:
        speaker_padded = None
        audio_frames_padded = None
        max_len = max(b["audio"].shape[0] for b in batch)
        audio_padded = torch.zeros(len(batch), max_len)
        audio_attention_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        for i, b in enumerate(batch):
            t = b["audio"].shape[0]
            audio_padded[i, :t] = b["audio"]
            audio_attention_mask[i, :t] = True

    # Look up or default prosody z-scores
    # Keys match utterance_id from each dataset's __getitem__ (audio file stem).
    # Generated by scripts/compute_prosody_stats.py. Falls back to zeros if not found.
    prosody_z = torch.zeros(len(batch), 3)
    if prosody_lookup is not None:
        for i, b in enumerate(batch):
            uid = b.get("utterance_id")
            if uid is not None and uid in prosody_lookup:
                prosody_z[i] = prosody_lookup[uid]

    speaker_ids = [b["speaker_id"] for b in batch]
    genders = [b.get("gender") for b in batch]

    # Conversation context for temporal / cross-attn modules
    conversation_ids = [
        b.get("conversation_id", b.get("utterance_id", f"single_{i}"))
        for i, b in enumerate(batch)
    ]
    turn_indices = [b.get("turn_index", 0) for b in batch]

    # Word-level divergence: MFA timestamps + token-word boundaries.
    # Loaded inline by each dataset's __getitem__ when textgrid_root is set.
    # Samples without word features get empty lists to preserve batch alignment.
    word_timestamps_raw = [b.get("word_timestamps") for b in batch]
    token_word_boundaries_raw = [b.get("token_word_boundaries") for b in batch]
    has_word_feats = any(x is not None for x in word_timestamps_raw)
    if has_word_feats:
        word_timestamps = [x if x is not None else [] for x in word_timestamps_raw]
        token_word_boundaries = [x if x is not None else [] for x in token_word_boundaries_raw]
    else:
        word_timestamps = None
        token_word_boundaries = None

    return {
        "audio": audio_padded,
        "speaker_embed": speaker_padded,
        "audio_frames": audio_frames_padded,
        "is_precomputed": is_precomputed,
        "audio_attention_mask": audio_attention_mask,
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "prosody_z": prosody_z,
        "conflict_binary": torch.stack([b["conflict_binary"] for b in batch]).float(),
        "conflict_type_labels": torch.stack([b["conflict_type_labels"] for b in batch]),
        "severity": torch.stack([b.get("severity", torch.tensor(0.0)) for b in batch]),
        "conversation_ids": conversation_ids,
        "turn_indices": turn_indices,
        "speaker_ids": speaker_ids,
        "genders": genders,
        "utterance_ids": [b.get("utterance_id", "") for b in batch],
        "word_timestamps": word_timestamps,
        "token_word_boundaries": token_word_boundaries,
        "dataset_names": [b.get("dataset_name", "unknown") for b in batch],
    }


def make_collate_fn(
    augmentor: Any = None,
    prosody_lookup: Optional[Dict[str, torch.Tensor]] = None,
):
    """Factory: returns a collate_fn with augmentation baked in via closure.

    Usage::

        from data.augmentation import AudioAugmentor
        train_collate = make_collate_fn(augmentor=AudioAugmentor())
        val_collate   = make_collate_fn()  # no augmentation
    """
    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return _collate_core(batch, augmentor=augmentor, prosody_lookup=prosody_lookup)
    return _collate


# Backwards-compatible default (no augmentation)
def conflictnet_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Default collate function without augmentation."""
    return _collate_core(batch, augmentor=None)

