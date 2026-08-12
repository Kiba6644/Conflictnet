"""Smoke tests for the serve package (no GPU, no real checkpoint needed)."""

from __future__ import annotations

import os
import tempfile

import pytest


class TestServeConfig:
    def test_defaults(self):
        from serve.config import ServeConfig
        cfg = ServeConfig()
        assert cfg.device == "cuda"
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8000
        assert cfg.embed_dim == 256
        assert cfg.lora_r == 16
        assert cfg.temporal_max_turns == 16

    def test_from_env(self):
        from serve.config import ServeConfig
        env = {
            "SERVE_DEVICE": "cpu",
            "SERVE_PORT": "9000",
            "SERVE_EMBED_DIM": "128",
            "SERVE_USE_SPEAKER_NORM": "false",
        }
        for k, v in env.items():
            os.environ[k] = v
        try:
            cfg = ServeConfig.from_env()
            assert cfg.device == "cpu"
            assert cfg.port == 9000
            assert cfg.embed_dim == 128
            assert cfg.use_speaker_norm is False
        finally:
            for k in env:
                os.environ.pop(k, None)

    def test_from_env_bool_true(self):
        from serve.config import ServeConfig
        os.environ["SERVE_USE_TEMPORAL"] = "true"
        try:
            cfg = ServeConfig.from_env()
            assert cfg.use_temporal is True
        finally:
            os.environ.pop("SERVE_USE_TEMPORAL", None)

    def test_from_env_cors_list(self):
        from serve.config import ServeConfig
        os.environ["SERVE_CORS_ORIGINS_LIST"] = "http://a.com,http://b.com"
        try:
            cfg = ServeConfig.from_env()
            assert cfg.cors_origins == ["http://a.com", "http://b.com"]
        finally:
            os.environ.pop("SERVE_CORS_ORIGINS_LIST", None)


class TestServeSchemas:
    def test_health_response(self):
        from serve.schemas import HealthResponse
        r = HealthResponse(status="ok", model_loaded=True, device="cpu")
        d = r.model_dump()
        assert d["status"] == "ok"
        assert d["model_loaded"] is True
        assert d["device"] == "cpu"

    def test_predict_response(self):
        from serve.schemas import PredictResponse
        r = PredictResponse(
            conflict=True,
            probs={"sarcasm": 0.9, "suppression": 0.1, "deception": 0.05},
            severity=0.8,
            predicted_type="sarcasm",
            fused_embed=[0.1] * 256,
        )
        d = r.model_dump()
        assert d["conflict"] is True
        assert d["predicted_type"] == "sarcasm"
        assert len(d["fused_embed"]) == 256

    def test_predict_response_no_embed(self):
        from serve.schemas import PredictResponse
        r = PredictResponse(
            conflict=False,
            probs={"sarcasm": 0.1, "suppression": 0.2, "deception": 0.05},
            severity=0.0,
            predicted_type="none",
            fused_embed=None,
        )
        assert r.fused_embed is None

    def test_batch_request_validation(self):
        from serve.schemas import PredictBatchRequest, BatchItem
        r = PredictBatchRequest(items=[
            BatchItem(audio=b"wav", text="hello"),
            BatchItem(audio=b"wav2", text="world", prosody_z=[1.0, 0.5, -0.3]),
        ])
        assert len(r.items) == 2
        assert r.items[0].text == "hello"
        assert r.items[1].prosody_z == [1.0, 0.5, -0.3]

    def test_batch_request_min_length(self):
        from serve.schemas import PredictBatchRequest
        with pytest.raises(Exception):
            PredictBatchRequest(items=[])

    def test_batch_request_max_length(self):
        from serve.schemas import PredictBatchRequest, BatchItem
        with pytest.raises(Exception):
            PredictBatchRequest(items=[BatchItem(audio=b"x", text="x")] * 65)

    def test_severity_bounds(self):
        from serve.schemas import PredictResponse
        with pytest.raises(Exception):
            PredictResponse(
                conflict=True,
                probs={"sarcasm": 0.5, "suppression": 0.5, "deception": 0.5},
                severity=-0.1,
                predicted_type="sarcasm",
            )
        with pytest.raises(Exception):
            PredictResponse(
                conflict=True,
                probs={"sarcasm": 0.5, "suppression": 0.5, "deception": 0.5},
                severity=1.1,
                predicted_type="sarcasm",
            )

    def test_error_response(self):
        from serve.schemas import ErrorResponse
        r = ErrorResponse(detail="Something went wrong")
        assert r.detail == "Something went wrong"


class TestServeApiHelpers:
    def test_parse_optional_json_valid(self):
        from serve.api import _parse_optional_json
        assert _parse_optional_json('[1, 2, 3]') == [1, 2, 3]
        assert _parse_optional_json('{"key": "val"}') == {"key": "val"}

    def test_parse_optional_json_none(self):
        from serve.api import _parse_optional_json
        assert _parse_optional_json(None) is None

    def test_parse_optional_json_invalid(self):
        from serve.api import _parse_optional_json
        from fastapi import HTTPException
        with pytest.raises(HTTPException, match="Invalid JSON"):
            _parse_optional_json("not valid json{{{")


class TestServeModel:
    """Tests for ServeModel — mocking to avoid GPU/model dependency."""

    def test_load_audio_mono_16k(self):
        import torch
        from serve.model import ServeModel
        from serve.config import ServeConfig

        cfg = ServeConfig(device="cpu")
        model = ServeModel(cfg)

        sr = 16000
        waveform = torch.sin(2 * 3.14159 * 440 * torch.arange(sr * 1) / sr)
        waveform = waveform.unsqueeze(0)  # mono

        buf = io_wave(waveform, sr)
        result = model._load_audio(buf.read())
        buf.close()
        assert result.ndim == 1
        assert result.shape[0] <= int(10.0 * sr)

    def test_load_audio_stereo_44k(self):
        import torch
        from serve.model import ServeModel
        from serve.config import ServeConfig

        cfg = ServeConfig(device="cpu")
        model = ServeModel(cfg)

        sr = 44100
        t = torch.arange(sr * 1) / sr
        waveform = torch.stack([torch.sin(2 * 3.14159 * 440 * t),
                                torch.sin(2 * 3.14159 * 880 * t)])
        buf = io_wave(waveform, sr)
        result = model._load_audio(buf.read())
        buf.close()
        assert result.ndim == 1

    def test_load_audio_truncates(self):
        import torch
        from serve.model import ServeModel
        from serve.config import ServeConfig

        cfg = ServeConfig(device="cpu")
        model = ServeModel(cfg)

        sr = 16000
        waveform = torch.randn(int(sr * 15))  # 15s > MAX_AUDIO_LEN
        buf = io_wave(waveform.unsqueeze(0), sr)
        result = model._load_audio(buf.read())
        buf.close()
        assert result.shape[0] <= int(10.0 * sr)  # truncated to 10s

    def test_tokenize(self):
        from serve.model import ServeModel
        from serve.config import ServeConfig
        from transformers import AutoTokenizer

        cfg = ServeConfig(device="cpu", max_text_len=64)
        model = ServeModel(cfg)
        model.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

        input_ids, attn_mask = model._tokenize("Hello world")
        assert input_ids.ndim == 1
        assert attn_mask.ndim == 1
        assert input_ids.shape[0] == 64  # padded to max_text_len
        assert attn_mask.shape[0] == 64

    def test_predict_requires_load(self):
        from serve.model import ServeModel
        from serve.config import ServeConfig

        cfg = ServeConfig(device="cpu")
        model = ServeModel(cfg)
        with pytest.raises(AssertionError, match="Model not loaded"):
            model.predict(audio_bytes=b"\x00", text="hello")

    def test_load_missing_checkpoint(self):
        from serve.model import ServeModel
        from serve.config import ServeConfig

        cfg = ServeConfig(checkpoint_path="/nonexistent/checkpoint.safetensors", device="cpu")
        model = ServeModel(cfg)
        with pytest.raises(FileNotFoundError):
            model.load()


def io_wave(waveform, sample_rate):
    """Write a torch tensor to a WAV BytesIO buffer."""
    import io
    import torchaudio

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torchaudio.save(tmp_path, waveform, sample_rate)
        with open(tmp_path, "rb") as f:
            buf = io.BytesIO(f.read())
        buf.seek(0)
        return buf
    finally:
        import os
        os.unlink(tmp_path)
