"""
engines/__init__.py
────────────────────
Engine Factory — reads the ASR_ENGINE environment variable and returns
the correct BaseASREngine implementation.

Usage (in lifespan)
-------------------
    from engines import create_engine
    engine = create_engine()
    engine.load_model()

Environment variables
---------------------
ASR_ENGINE      : "nemotron" (default) | "qwen"
MODEL_PATH      : local path for NeMo model (NemotronEngine only)
QWEN_MODEL_ID   : HF model ID for the Qwen engine (QwenHuggingFaceEngine only)
QWEN_LANGUAGE   : ISO 639-1 language code for Qwen (default: "ar")
QWEN_CTX_WINDOW_S : seconds of audio context for Qwen window (default: 8)
"""

from __future__ import annotations

import logging
import os

from .base import BaseASREngine

log = logging.getLogger("asr_server.engine_factory")

# ── Known engine names and their aliases ────────────────────────────────────
_NEMOTRON_NAMES  = {"nemotron", "nemo", "ara-nemotron"}
_QWEN_HF_NAMES   = {"qwen", "qwen-hf", "huggingface", "hf", "sensevoice"}


def create_engine(engine_name: str | None = None) -> BaseASREngine:
    """
    Instantiate and return the requested ASR engine.

    Parameters
    ----------
    engine_name : str | None
        Engine identifier.  If None, reads ``ASR_ENGINE`` env var.
        Defaults to ``"nemotron"`` if neither is supplied.

    Returns
    -------
    BaseASREngine
        The requested engine (model NOT yet loaded — call ``.load_model()``).

    Raises
    ------
    ValueError
        If ``engine_name`` is not recognised.
    """
    name = (engine_name or os.getenv("ASR_ENGINE", "nemotron")).strip().lower()

    log.info("Engine factory: requested engine = %r", name)

    if name in _NEMOTRON_NAMES:
        from .nemotron import NemotronEngine
        engine = NemotronEngine(
            model_path  = os.getenv("MODEL_PATH", "./model"),
            model_hf_id = os.getenv(
                "NEMOTRON_MODEL_HF_ID",
                "Abdelkareem/Ara-nemotron-3.5-asr-streaming-0.6b",
            ),
        )
        log.info("Engine factory: selected NemotronEngine (NeMo RNNT)")
        return engine

    elif name in _QWEN_HF_NAMES:
        from .qwen_hf import QwenHuggingFaceEngine
        import os as _os
        engine = QwenHuggingFaceEngine(
            model_id     = _os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3-ASR-0.6B"),
            language     = _os.getenv("QWEN_LANGUAGE", "ar") or None,
            ctx_window_s = int(_os.getenv("QWEN_CTX_WINDOW_S", "8")),
        )
        log.info("Engine factory: selected QwenHuggingFaceEngine (HF Seq2Seq)")
        return engine

    else:
        raise ValueError(
            f"Unknown ASR_ENGINE value: {name!r}. "
            f"Valid options: {sorted(_NEMOTRON_NAMES | _QWEN_HF_NAMES)}"
        )


__all__ = ["create_engine", "BaseASREngine"]
