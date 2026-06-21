"""
engines/qwen_hf.py
───────────────────
Qwen3-ASR Engine — Engine B.

Uses the official `qwen-asr` Python package (pip install qwen-asr)
with its `Qwen3ASRModel` class, which is the correct and supported way
to run Qwen3-ASR-0.6B / 1.7B models.

Streaming Strategy
------------------
Qwen3-ASR is an offline model. We simulate streaming by running
inference on a rolling window (up to `ctx_window_s` seconds) on every
call, matching what StreamingSession already does for NeMo.

Install requirements:
    pip install -U qwen-asr
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch

from .base import BaseASREngine

log = logging.getLogger("asr_server.qwen_hf")

# ── Configuration ─────────────────────────────────────────────────────────────
_DEFAULT_QWEN_MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3-ASR-0.6B")
_SAMPLE_RATE           = 16_000   # Hz — must match app-level SAMPLE_RATE

# Maximum audio window (seconds) fed into Qwen on each inference call.
# Qwen3-ASR handles up to 30 s; 8 s balances latency vs. context.
_CTX_WINDOW_S = int(os.getenv("QWEN_CTX_WINDOW_S", "8"))

# Language to pass to transcribe(). Set to None for auto-detect.
# Full language names are required by qwen-asr (e.g. "Arabic", not "ar").
# Mapping: ISO codes → full names supported by qwen-asr
_ISO_TO_FULL = {
    "ar": "Arabic",   "zh": "Chinese",  "en": "English",
    "yue": "Cantonese", "de": "German",  "fr": "French",
    "es": "Spanish",  "pt": "Portuguese", "id": "Indonesian",
    "it": "Italian",  "ko": "Korean",  "ru": "Russian",
    "th": "Thai",     "vi": "Vietnamese", "ja": "Japanese",
    "tr": "Turkish",  "hi": "Hindi",   "ms": "Malay",
    "nl": "Dutch",    "sv": "Swedish",  "da": "Danish",
    "fi": "Finnish",  "pl": "Polish",   "cs": "Czech",
    "fil": "Filipino", "fa": "Persian",  "el": "Greek",
    "ro": "Romanian", "hu": "Hungarian", "mk": "Macedonian",
}

def _normalize_language(lang: str | None) -> str | None:
    """Convert ISO code (e.g. 'ar') to full name ('Arabic') for qwen-asr."""
    if lang is None:
        return None
    return _ISO_TO_FULL.get(lang.lower(), lang)  # fallback: pass as-is

_LANGUAGE = _normalize_language(os.getenv("QWEN_LANGUAGE", "Arabic"))


class QwenHuggingFaceEngine(BaseASREngine):
    """
    Qwen3-ASR engine using the official `qwen-asr` package.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. ``"Qwen/Qwen3-ASR-0.6B"``.
    language : str | None
        Full language name (e.g. "Arabic", "English") or None for auto-detect.
    ctx_window_s : int
        Rolling window in seconds of audio fed per inference call.
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_QWEN_MODEL_ID,
        language: Optional[str] = _LANGUAGE,
        ctx_window_s: int = _CTX_WINDOW_S,
    ) -> None:
        self.model_id      = model_id
        self.language      = _normalize_language(language)   # always full name
        self.ctx_window_s  = ctx_window_s
        self._ctx_samples  = ctx_window_s * _SAMPLE_RATE

        # Set after load_model()
        self._model: Optional[object] = None

    # ──────────────────────────────────────────────────────────────────────────
    # BaseASREngine interface
    # ──────────────────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """
        Load Qwen3-ASR via the official `qwen-asr` package.
        Requires: pip install -U qwen-asr
        """
        try:
            from qwen_asr import Qwen3ASRModel  # official package
        except ImportError:
            log.exception(
                "QwenHFEngine: 'qwen-asr' package not installed!\n"
                "  Fix: pip install -U qwen-asr"
            )
            raise

        use_cuda = torch.cuda.is_available()
        dtype    = torch.bfloat16 if use_cuda else torch.float32
        device   = "cuda:0" if use_cuda else "cpu"

        log.info(
            "  QwenHFEngine: loading %s  dtype=%s  device=%s …",
            self.model_id, dtype, device,
        )

        try:
            self._model = Qwen3ASRModel.from_pretrained(
                self.model_id,
                dtype=dtype,
                device_map=device,
                max_inference_batch_size=8,
                max_new_tokens=256,
            )
        except Exception:
            log.exception("CRITICAL — QwenHFEngine.load_model() raised an unhandled exception.")
            raise

        if use_cuda:
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            log.info(
                "  QwenHFEngine ready ✓  model=%s  lang=%s  ctx=%ds  GPU=%s (%.1f GB VRAM)",
                self.model_id, self.language, self.ctx_window_s,
                torch.cuda.get_device_name(0), vram,
            )
        else:
            log.warning("  QwenHFEngine: CUDA unavailable — running on CPU (slow)")

    # ──────────────────────────────────────────────────────────────────────────

    def run_inference(self, audio_context: np.ndarray) -> str:
        """
        Run Qwen3-ASR on the most recent `ctx_window_s` seconds of audio.

        Parameters
        ----------
        audio_context : np.ndarray
            Float32 PCM samples, mono 16 kHz. Full accumulated context
            window from StreamingSession (up to 10 s).
        """
        if self._model is None:
            log.error("QwenHFEngine: model not loaded — call load_model() first")
            return ""

        try:
            # ── Sliding window: use the last ctx_window_s seconds ────────────
            if len(audio_context) > self._ctx_samples:
                window = audio_context[-self._ctx_samples:]
            else:
                window = audio_context

            # Ensure float32 for the package (it handles internal conversion)
            window_f32 = window.astype(np.float32)

            # ── Transcribe ───────────────────────────────────────────────────
            # qwen-asr accepts (np.ndarray, sample_rate) tuple directly
            results = self._model.transcribe(
                audio=(window_f32, _SAMPLE_RATE),
                language=self.language,  # None = auto-detect
            )

            text = results[0].text.strip() if results else ""
            if text:
                log.info("🎯 Qwen ASR: %s", text)
            return text

        except Exception as exc:
            log.error("QwenHFEngine inference error: %s", exc, exc_info=True)
            return ""

    # ──────────────────────────────────────────────────────────────────────────

    def reset_state(self) -> None:
        """
        Reset per-utterance state.
        Qwen3-ASR is stateless across calls — no-op.
        """
        pass

    @property
    def name(self) -> str:
        return f"Qwen3ASR({self.model_id.split('/')[-1]})"
