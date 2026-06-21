"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Ara-Nemotron Streaming ASR — FastAPI Backend
  Model : Abdelkareem/Ara-nemotron-3.5-asr-streaming-0.6b
  Stack : FastAPI · NeMo · CUDA (RTX 3080)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 1 — Force safe torchaudio backend + enable C-level crash reporting.
#
# (a) TORCHAUDIO_BACKEND=soundfile  — pure-Python I/O, no native DLL loaded.
#     In torchaudio >= 2.1 the dispatcher is always on; set_audio_backend() is
#     a no-op, so we must use the env var instead.
# (b) TORCHAUDIO_USE_BACKEND_DISPATCHER=1 — kept for older torchaudio compat.
# (c) faulthandler — writes a C-level stack trace to stderr on segfault/SIGABRT
#     so the crash is never completely silent again.
# ─────────────────────────────────────────────────────────────────────────────
import os
import faulthandler
import sys

# Enable native crash reporting (writes to stderr on segfault / access violation)
faulthandler.enable(file=sys.stderr, all_threads=True)

# Backend env vars must be set BEFORE torchaudio is imported by any library
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "1"
os.environ["TORCHAUDIO_BACKEND"] = "soundfile"   # works in torchaudio >= 2.1

# ── PATCH 4 — Redirect temp dir to D:\ to avoid "No space left on device" ──
# NeMo extracts .nemo model files (tar archives) to a temp directory.
# On Windows, the default temp is C:\Users\...\AppData\Local\Temp which can
# fill up quickly with large CUDA/NeMo installs.  Force it to D:\tmp which
# has more free space.  Must be set before any NeMo / tempfile import.
_NEMO_TMPDIR = r"D:\tmp"
os.makedirs(_NEMO_TMPDIR, exist_ok=True)
os.environ["TEMP"]   = _NEMO_TMPDIR
os.environ["TMP"]    = _NEMO_TMPDIR
os.environ["TMPDIR"] = _NEMO_TMPDIR
import tempfile as _tempfile
_tempfile.tempdir = _NEMO_TMPDIR

import warnings
import torchaudio  # noqa: E402
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    try:
        torchaudio.set_audio_backend("soundfile")   # no-op in >= 2.1, harmless
    except Exception:
        pass  # dispatcher already handles it via TORCHAUDIO_BACKEND env var

import asyncio
import json
import logging
import math
import time
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("asr_server")

# Suppress noisy NeMo / PyTorch logs
for lib in ("nemo", "nemo_logger", "lightning", "torch.distributed"):
    logging.getLogger(lib).setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (override via environment variables)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH    = os.getenv("MODEL_PATH", "./model")          # local dir or .nemo file
MODEL_HF_ID   = "Abdelkareem/Ara-nemotron-3.5-asr-streaming-0.6b"
SAMPLE_RATE   = 16_000                 # Hz — model expects 16 kHz mono PCM

# Silence / endpointing
SILENCE_RMS_THRESHOLD  = float(os.getenv("SILENCE_RMS", "0.007"))
SILENCE_FRAMES_TRIGGER = int(os.getenv("SILENCE_FRAMES", "6"))   # ×chunk_ms → ~1.5 s

# Thread pool for CPU-bound inference
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr_infer")


# ─────────────────────────────────────────────────────────────────────────────
# Global model (loaded once at startup — stays resident on GPU)
# ─────────────────────────────────────────────────────────────────────────────
asr_model: object = None


def _load_model() -> object:
    """
    Load NeMo ASR model from a local .nemo file, a local dir, or HuggingFace.

    Loads as EncDecHybridRNNTCTCBPEModelWithPrompt (aliased via the compat stub
    as EncDecRNNTBPEModelWithPrompt) so that transcribe() supports the
    target_lang kwarg needed to force Arabic output from this multilingual model.
    strict=False lets the missing aux-CTC weights be safely skipped.

    PATCH 2 — The entire function body is wrapped in try/except so that any
    error (ImportError, FileNotFoundError, CUDA OOM, C++ exception bubbling up
    through ctypes, etc.) is logged with a full traceback BEFORE the thread
    dies.  Without this, exceptions raised inside a ThreadPoolExecutor are
    silently swallowed on Windows and the process exits without any output.
    """
    try:
        import nemo.collections.asr as nemo_asr   # deferred — heavy import
        from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models_prompt import (
            EncDecHybridRNNTCTCBPEModelWithPrompt,
        )

        local = Path(MODEL_PATH)

        def _restore(path: str):
            """Restore with strict=False and WithPrompt class for target_lang support."""
            log.info("  restore_from (WithPrompt, strict=False): %s", path)
            try:
                model = EncDecHybridRNNTCTCBPEModelWithPrompt.restore_from(
                    restore_path=path,
                    map_location="cpu",
                    strict=False,
                )
                log.info("  Loaded as EncDecHybridRNNTCTCBPEModelWithPrompt (strict=False) ✔")
                return model
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
            log.info("  Loading from .nemo file: %s", local.resolve())
            return _restore(str(local))

        # 2) Directory containing a .nemo file
        if local.is_dir():
            nemo_files = sorted(local.glob("*.nemo"))
            if nemo_files:
                log.info("  Loading .nemo from dir: %s", nemo_files[0])
                return _restore(str(nemo_files[0]))

        # 3) Fall back to HuggingFace / NeMo NGC cache
        log.info("  Downloading from HuggingFace: %s", MODEL_HF_ID)
        return nemo_asr.models.ASRModel.from_pretrained(MODEL_HF_ID, map_location="cpu")

    except Exception as e:  # noqa: BLE001
        # ── CRITICAL: log full traceback so the real error is never hidden ──
        log.exception(
            "CRITICAL STARTUP ERROR — _load_model() raised an unhandled exception.\n"
            "The server cannot continue.  Real error:\n"
        )
        # Re-raise so the executor future carries the exception and lifespan
        # can surface it (see PATCH 3 in lifespan below).
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    PATCH 3 — Hardened lifespan that guarantees any startup failure is
    printed with a full traceback before the process dies.

    Problem on Windows: if run_in_executor raises, the Future exception is
    stored but asyncio may swallow it when the event-loop tears down, giving
    the infamous "silent crash".  We explicitly await + re-raise inside a
    try/except so log.exception() always fires first.
    """
    global asr_model

    log.info("═" * 62)
    log.info("  Ara-Nemotron Streaming ASR  —  server starting…")
    log.info("═" * 62)
    log.info("  torchaudio backend : %s", torchaudio.get_audio_backend())
    log.info("  torch version      : %s", torch.__version__)
    log.info("  CUDA available     : %s", torch.cuda.is_available())

    try:
        # Load in a thread so the event-loop isn't blocked.
        # We store the Future explicitly so we can inspect its exception.
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(_EXECUTOR, _load_model)

        # Await with a generous timeout so a hung load doesn't silently block.
        try:
            asr_model = await asyncio.wait_for(future, timeout=600)  # 10 min max
        except asyncio.TimeoutError:
            log.exception(
                "CRITICAL STARTUP ERROR — _load_model() timed out after 600 s.\n"
                "Check for a hung download or a deadlock in NeMo initialisation."
            )
            raise RuntimeError("Model load timed out") from None

        if asr_model is None:
            raise RuntimeError("_load_model() returned None — check logs above")

        asr_model.eval()

        # Move to GPU if available
        if torch.cuda.is_available():
            asr_model = asr_model.cuda()
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            log.info("  GPU  : %s (%.1f GB VRAM)", torch.cuda.get_device_name(0), vram)
        else:
            log.warning("  CUDA unavailable — running on CPU (expect high latency)")

        log.info("  Model ready  ✓")
        log.info("═" * 62)

    except Exception as e:  # noqa: BLE001
        # ── Guarantee the real error is visible before the process exits ──
        log.exception(
            "CRITICAL STARTUP ERROR — lifespan failed.  "
            "Server will NOT start.  Real error:\n"
        )
        # Re-raise so uvicorn/starlette knows startup failed (exits non-zero)
        raise

    yield
    log.info("Server shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Ara-Nemotron Streaming ASR",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def serve_ui():
    return FileResponse("index.html")


@app.get("/health")
async def health():
    """Quick liveness + model-info endpoint."""
    return {
        "status"      : "ok",
        "model_loaded": asr_model is not None,
        "cuda"        : torch.cuda.is_available(),
        "device"      : torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "model_id"    : MODEL_HF_ID,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-Connection Streaming Session
# ─────────────────────────────────────────────────────────────────────────────
# How many samples to accumulate before re-running transcription (0.5 s)
_TRANSCRIBE_EVERY_N_SAMPLES = SAMPLE_RATE // 2   # 8 000 @ 16 kHz
# Maximum audio history kept for context (10 s)
_MAX_CONTEXT_SAMPLES = SAMPLE_RATE * 10


class StreamingSession:
    """
    Direct-tensor streaming ASR session — bypasses Lhotse/manifest entirely.

    Flow per chunk:
      raw audio (float32) → preprocessor (mel) → encoder
      → Arabic prompt injection (one-hot index-7 + prompt_kernel)
      → RNNT greedy decoder → partial / final text

    Audio is accumulated and re-inferred every ~0.5 s on the last ≤10 s.
    Sustained silence triggers a "final" event and resets the buffer.
    """

    # Arabic language ID in the Nemotron prompt_dictionary
    _ARABIC_PROMPT_IDX = 7

    def __init__(self, model):
        self.model   = model
        self.device  = next(model.parameters()).device
        self._chunks: list[np.ndarray] = []
        self._pending_samples = 0
        self.silence_frames   = 0
        self.last_partial     = ""
        self.chunk_count      = 0

        # Derive num_prompts from model config (default 128)
        self.num_prompts = int(
            getattr(model, 'num_prompts', None)
            or model.cfg.model_defaults.get('num_prompts', 128)
        )
        self.has_prompt = getattr(model, 'concat', False) and hasattr(model, 'prompt_kernel')
        log.info("StreamingSession init — prompt_kernel=%s  num_prompts=%d  device=%s",
                 self.has_prompt, self.num_prompts, self.device)

    # ── Reset ───────────────────────────────────────────────────────────────
    def _reset_for_next_utterance(self):
        self._chunks          = []
        self._pending_samples = 0
        self.last_partial     = ""
        self.silence_frames   = 0

    # ── Utility ─────────────────────────────────────────────────────────────
    @staticmethod
    def _rms(samples: np.ndarray) -> float:
        return float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0

    # ── Core inference  (runs in ThreadPoolExecutor) ──────────────────────────
    def _run_inference(self, samples: np.ndarray) -> str:
        """
        Direct tensor forward pass — NO temp files, NO Lhotse, NO manifest.

        Pipeline:
            raw audio tensor [1,T]
            → model.preprocessor  → mel [1,D,T']
            → model.encoder       → encoded [1,C,T'']
            → transpose           → [1,T'',C]
            → concat + prompt_kernel (Arabic=idx 7) if supported
            → transpose back      → [1,C,T'']
            → decoding.rnnt_decoder_predictions_tensor
            → text
        """
        try:
            self._chunks.append(samples)
            self._pending_samples += len(samples)

            # Wait until 0.5 s of new audio is accumulated
            if self._pending_samples < _TRANSCRIBE_EVERY_N_SAMPLES:
                return self.last_partial

            self._pending_samples = 0

            # Build context window (last ≤10 s)
            ctx = np.concatenate(self._chunks)
            if len(ctx) > _MAX_CONTEXT_SAMPLES:
                ctx = ctx[-_MAX_CONTEXT_SAMPLES:]
                self._chunks = [ctx]

            # ── Prepare tensors ───────────────────────────────────────
            audio_t   = torch.from_numpy(ctx).float().unsqueeze(0).to(self.device)   # [1,T]
            audio_len = torch.tensor([ctx.shape[0]], dtype=torch.long, device=self.device)

            with torch.inference_mode():
                # Disable NeMo neural-type checks (safe for inference)
                from nemo.core.classes.common import typecheck
                with typecheck.disable_checks():
                    # 1. Mel-spectrogram features
                    processed, proc_len = self.model.preprocessor(
                        input_signal=audio_t, length=audio_len
                    )

                    # 2. Encoder: [1,D,T'] → encoded [1,C,T'']
                    encoded, enc_len = self.model.encoder(
                        audio_signal=processed, length=proc_len
                    )

                # 3. Transpose for prompt injection: [1,C,T''] → [1,T'',C]
                enc_t = encoded.transpose(1, 2)

                # 4. Arabic prompt conditioning via prompt_kernel
                if self.has_prompt:
                    T_enc = enc_t.shape[1]
                    prompt = torch.zeros(
                        1, T_enc, self.num_prompts,
                        dtype=enc_t.dtype, device=self.device
                    )
                    prompt[:, :, self._ARABIC_PROMPT_IDX] = 1.0   # ar-AR = 7
                    enc_t = self.model.prompt_kernel(
                        torch.cat([enc_t, prompt], dim=-1)
                    ).to(encoded.dtype)

                # 5. Transpose back: [1,T'',C] → [1,C,T'']
                encoded_final = enc_t.transpose(1, 2)

                # 6. RNNT greedy decode — returns list[str]
                best_hyp = self.model.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=encoded_final,
                    encoded_lengths=enc_len,
                    return_hypotheses=False,
                )

            if best_hyp:
                h = best_hyp[0]
                text = (h.text if hasattr(h, "text") else str(h)).strip()
                if text:
                    log.info("🎯 النص: %s", text)
                return text

            return self.last_partial

        except Exception as exc:
            log.error("Inference error: %s", exc, exc_info=True)
            return self.last_partial   # keep last known good text

    # ── Main entry-point: called from WebSocket handler ───────────────────────
    def process(self, pcm_bytes: bytes) -> dict | None:
        """
        Process one raw PCM chunk (int16-LE · mono · 16 kHz).
        Returns a JSON-ready dict or None when nothing changed.

        Response schema
        ───────────────
        {"type": "partial", "text": "...", "latency_ms": 42.1}
        {"type": "final",   "text": "...", "latency_ms": 0   }
        """
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return None

        self.chunk_count += 1
        is_silent = self._rms(samples) < SILENCE_RMS_THRESHOLD
        partial   = self._run_inference(samples)

        self.silence_frames = (self.silence_frames + 1) if is_silent else 0

        # Endpointing — sustained silence → emit final
        if self.silence_frames >= SILENCE_FRAMES_TRIGGER and self.last_partial:
            final = self.last_partial
            self._reset_for_next_utterance()
            return {"type": "final", "text": final}

        # Emit partial only when it changes
        if partial and partial != self.last_partial:
            self.last_partial = partial
            return {"type": "partial", "text": partial}

        return None

    def flush(self) -> dict | None:
        """Emit any pending partial as final (called on client 'end' signal)."""
        if self.last_partial:
            final = self.last_partial
            self._reset_for_next_utterance()
            return {"type": "final", "text": final}
        return None


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket Endpoint  /ws/transcribe
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    await ws.accept()
    client_id = f"{ws.client.host}:{ws.client.port}"
    log.info("Client connected   %s", client_id)

    loop    = asyncio.get_event_loop()
    session = StreamingSession(asr_model)

    try:
        while True:
            # Receive the next frame (binary audio OR text control message)
            data = await ws.receive()

            # ── Graceful disconnect detection ──────────────────────────────
            if data.get("type") == "websocket.disconnect":
                log.info("Client disconnected  %s", client_id)
                break

            # ── Binary audio chunk ─────────────────────────────────────────
            if "bytes" in data and data["bytes"]:
                t0 = time.perf_counter()

                result = await loop.run_in_executor(
                    _EXECUTOR, session.process, data["bytes"]
                )

                latency_ms = round((time.perf_counter() - t0) * 1000, 1)

                if result is not None:
                    result["latency_ms"] = latency_ms
                    await ws.send_json(result)

            # ── Control / text message ─────────────────────────────────────
            elif "text" in data and data["text"]:
                try:
                    msg = json.loads(data["text"])
                except json.JSONDecodeError:
                    continue

                if msg.get("type") == "end":
                    # Client stopped recording — flush pending partial
                    final = session.flush()
                    if final:
                        final["latency_ms"] = 0
                        await ws.send_json(final)
                    await ws.send_json({"type": "eos"})   # end-of-stream signal
                    break

                elif msg.get("type") == "ping":
                    await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        log.info("Client disconnected  %s", client_id)
    except Exception as exc:
        log.exception("WS error for %s: %s", client_id, exc)
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Dev entry-point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
