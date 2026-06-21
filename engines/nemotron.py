"""
engines/nemotron.py
────────────────────
NeMo ASR Engine — Engine A.

Wraps the existing Ara-Nemotron streaming model logic that was previously
embedded directly in app.py / StreamingSession._run_inference().

Model : Abdelkareem/Ara-nemotron-3.5-asr-streaming-0.6b
Stack : NVIDIA NeMo · CUDA

Pipeline (per inference call):
    raw audio [float32, mono, 16 kHz]
    → model.preprocessor  → mel spectrogram [1, D, T']
    → model.encoder       → encoded [1, C, T'']
    → (optional) prompt_kernel  (Arabic lang conditioning, idx=7)
    → decoding.rnnt_decoder_predictions_tensor
    → text string
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .base import BaseASREngine

log = logging.getLogger("asr_server.nemotron")

# ── Arabic language prompt index in the Nemotron prompt_dictionary ──────────
_ARABIC_PROMPT_IDX = 7

# Default HuggingFace model ID (used as fallback when local model is absent)
_DEFAULT_HF_ID = "Abdelkareem/Ara-nemotron-3.5-asr-streaming-0.6b"


class NemotronEngine(BaseASREngine):
    """
    NVIDIA NeMo Hybrid-RNNT/CTC streaming ASR engine.

    Parameters
    ----------
    model_path : str
        Path to a local .nemo file *or* a directory containing one.
        Falls back to ``model_hf_id`` if no local file is found.
    model_hf_id : str
        HuggingFace / NeMo NGC model ID used as the remote fallback.
    """

    def __init__(
        self,
        model_path: str = "./model",
        model_hf_id: str = _DEFAULT_HF_ID,
    ) -> None:
        self.model_path  = model_path
        self.model_hf_id = model_hf_id

        # Set after load_model()
        self._model: Optional[object] = None
        self._device: Optional[torch.device] = None
        self._num_prompts: int = 128
        self._has_prompt: bool = False

    # ────────────────────────────────────────────────────────────────────────
    # BaseASREngine interface
    # ────────────────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """
        Load NeMo ASR model from a local .nemo file, a local dir, or HF.

        Uses EncDecHybridRNNTCTCBPEModelWithPrompt so that transcribe()
        supports the target_lang kwarg needed to force Arabic output from
        this multilingual model.  strict=False lets missing aux-CTC weights
        be safely skipped.

        NOTE: This is intentionally wrapped in a broad try/except so that
        any error (ImportError, CUDA OOM, C++ exception, etc.) is logged
        with a full traceback BEFORE the thread pool dies silently on Windows.
        """
        try:
            import nemo.collections.asr as nemo_asr   # deferred — heavy import
            from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models_prompt import (
                EncDecHybridRNNTCTCBPEModelWithPrompt,
            )

            local = Path(self.model_path)

            def _restore(path: str):
                """Restore with strict=False + WithPrompt for target_lang support."""
                log.info("  restore_from (WithPrompt, strict=False): %s", path)
                try:
                    m = EncDecHybridRNNTCTCBPEModelWithPrompt.restore_from(
                        restore_path=path,
                        map_location="cpu",
                        strict=False,
                    )
                    log.info("  Loaded as EncDecHybridRNNTCTCBPEModelWithPrompt ✔")
                    return m
                except Exception as e1:
                    log.warning("  WithPrompt load failed: %s", e1)
                    log.warning("  Falling back to base ASRModel (no language prompt support)")
                    return nemo_asr.models.ASRModel.restore_from(
                        restore_path=path,
                        map_location="cpu",
                        strict=False,
                    )

            # 1) Explicit .nemo file
            if local.is_file() and local.suffix == ".nemo":
                log.info("  NemotronEngine: loading from .nemo file: %s", local.resolve())
                model = _restore(str(local))

            # 2) Directory containing a .nemo file
            elif local.is_dir():
                nemo_files = sorted(local.glob("*.nemo"))
                if nemo_files:
                    log.info("  NemotronEngine: loading from dir: %s", nemo_files[0])
                    model = _restore(str(nemo_files[0]))
                else:
                    # No .nemo found in dir → HF fallback
                    log.info("  NemotronEngine: downloading from HuggingFace: %s", self.model_hf_id)
                    model = nemo_asr.models.ASRModel.from_pretrained(
                        self.model_hf_id, map_location="cpu"
                    )
            else:
                # 3) Fallback to HuggingFace / NeMo NGC cache
                log.info("  NemotronEngine: downloading from HuggingFace: %s", self.model_hf_id)
                model = nemo_asr.models.ASRModel.from_pretrained(
                    self.model_hf_id, map_location="cpu"
                )

            model.eval()

            # Move to GPU
            if torch.cuda.is_available():
                model = model.cuda()
                vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                log.info(
                    "  NemotronEngine: GPU=%s  (%.1f GB VRAM)",
                    torch.cuda.get_device_name(0), vram,
                )
            else:
                log.warning("  NemotronEngine: CUDA unavailable — running on CPU")

            self._model  = model
            self._device = next(model.parameters()).device

            # Derive prompt metadata from model config (default 128)
            self._num_prompts = int(
                getattr(model, "num_prompts", None)
                or model.cfg.model_defaults.get("num_prompts", 128)
            )
            self._has_prompt = (
                getattr(model, "concat", False) and hasattr(model, "prompt_kernel")
            )

            log.info(
                "  NemotronEngine ready ✓  prompt_kernel=%s  num_prompts=%d  device=%s",
                self._has_prompt, self._num_prompts, self._device,
            )

        except Exception:
            log.exception(
                "CRITICAL — NemotronEngine.load_model() raised an unhandled exception."
            )
            raise

    # ────────────────────────────────────────────────────────────────────────

    def run_inference(self, audio_context: np.ndarray) -> str:
        """
        Run the NeMo RNNT pipeline on the given float32 audio context.

        Pipeline:
            audio tensor [1,T]
            → preprocessor  → mel [1,D,T']
            → encoder       → encoded [1,C,T'']
            → transpose     → [1,T'',C]
            → prompt_kernel (Arabic idx=7) if supported
            → transpose     → [1,C,T'']
            → rnnt_decoder_predictions_tensor
            → text
        """
        if self._model is None:
            log.error("NemotronEngine: model not loaded — call load_model() first")
            return ""

        try:
            audio_t   = torch.from_numpy(audio_context).float().unsqueeze(0).to(self._device)
            audio_len = torch.tensor([audio_context.shape[0]], dtype=torch.long, device=self._device)

            with torch.inference_mode():
                from nemo.core.classes.common import typecheck
                with typecheck.disable_checks():
                    # 1. Mel-spectrogram features
                    processed, proc_len = self._model.preprocessor(
                        input_signal=audio_t, length=audio_len
                    )

                    # 2. Encoder: [1,D,T'] → encoded [1,C,T'']
                    encoded, enc_len = self._model.encoder(
                        audio_signal=processed, length=proc_len
                    )

                # 3. Transpose for prompt injection: [1,C,T''] → [1,T'',C]
                enc_t = encoded.transpose(1, 2)

                # 4. Arabic prompt conditioning via prompt_kernel
                if self._has_prompt:
                    T_enc = enc_t.shape[1]
                    prompt = torch.zeros(
                        1, T_enc, self._num_prompts,
                        dtype=enc_t.dtype, device=self._device,
                    )
                    prompt[:, :, _ARABIC_PROMPT_IDX] = 1.0   # ar-AR = 7
                    enc_t = self._model.prompt_kernel(
                        torch.cat([enc_t, prompt], dim=-1)
                    ).to(encoded.dtype)

                # 5. Transpose back: [1,T'',C] → [1,C,T'']
                encoded_final = enc_t.transpose(1, 2)

                # 6. RNNT greedy decode → list[str | Hypothesis]
                best_hyp = self._model.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=encoded_final,
                    encoded_lengths=enc_len,
                    return_hypotheses=False,
                )

            if best_hyp:
                h = best_hyp[0]
                text = (h.text if hasattr(h, "text") else str(h)).strip()
                if text:
                    log.info("🎯 النص (NeMo): %s", text)
                return text

            return ""

        except Exception as exc:
            log.error("NemotronEngine inference error: %s", exc, exc_info=True)
            return ""

    # ────────────────────────────────────────────────────────────────────────

    def reset_state(self) -> None:
        """NeMo is stateless per call — nothing to reset."""
        pass

    @property
    def name(self) -> str:
        return "NemotronEngine"

    @property
    def model_id(self) -> str:
        return self.model_hf_id
