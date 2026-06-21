"""
engines/base.py
───────────────
Abstract Base Class that every ASR engine must implement.

Any concrete engine (NeMo, HuggingFace, etc.) must subclass
BaseASREngine and implement the three abstract methods below.
"""

from __future__ import annotations

import abc
import numpy as np


class BaseASREngine(abc.ABC):
    """
    Strategy interface for ASR engines.

    Lifecycle
    ---------
    1. Instantiate the engine (pass config via __init__).
    2. Call ``load_model()`` once at startup (inside lifespan).
    3. For every inference call: ``run_inference(audio_context_array)``.
    4. Call ``reset_state()`` between utterances if the engine holds state.
    """

    # ── Required methods ────────────────────────────────────────────────────

    @abc.abstractmethod
    def load_model(self) -> None:
        """
        Load and warm-up the model.
        Called once in the FastAPI lifespan manager.
        Must move the model to GPU if CUDA is available.
        """
        ...

    @abc.abstractmethod
    def run_inference(self, audio_context: np.ndarray) -> str:
        """
        Run ASR inference on the given audio context.

        Parameters
        ----------
        audio_context : np.ndarray
            Float32 PCM samples, mono, 16 kHz.
            This is the *full accumulated context window* (up to 10 s),
            not just the latest chunk.

        Returns
        -------
        str
            Transcribed text (may be an empty string if nothing detected).
        """
        ...

    @abc.abstractmethod
    def reset_state(self) -> None:
        """
        Reset any internal per-utterance state.
        Called after a 'final' event is emitted (silence endpoint triggered
        or client sends 'end').  Stateless engines can implement as a no-op.
        """
        ...

    # ── Optional helpers ────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Human-readable engine identifier (used in /health response)."""
        return self.__class__.__name__
