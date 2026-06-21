"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Dual-Engine Streaming ASR — FastAPI Backend
  Engine A : Ara-Nemotron (NVIDIA NeMo)
  Engine B : Qwen3-ASR    (HuggingFace Transformers)
  Stack    : FastAPI · WebSocket · CUDA (RTX 3080)

  Select engine via: ASR_ENGINE=nemotron | qwen
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
import time
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import FileResponse

from engines import BaseASREngine, create_engine

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
SAMPLE_RATE = 16_000                # Hz — both models expect 16 kHz mono PCM

# Which engine to use: "nemotron" (default) or "qwen"
ASR_ENGINE_NAME = os.getenv("ASR_ENGINE", "nemotron")

# Silence / endpointing
SILENCE_RMS_THRESHOLD  = float(os.getenv("SILENCE_RMS", "0.007"))
SILENCE_FRAMES_TRIGGER = int(os.getenv("SILENCE_FRAMES", "6"))   # ×chunk_ms → ~1.5 s

# Thread pool for CPU-bound inference
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr_infer")


# ─────────────────────────────────────────────────────────────────────────────
# Global engine instance (loaded once at startup — stays resident on GPU)
# ─────────────────────────────────────────────────────────────────────────────
asr_engine: BaseASREngine | None = None
_current_engine_key: str = ASR_ENGINE_NAME   # tracks active engine name string
_engine_switching: bool = False              # True while a hot-swap is in progress
_engine_lock = asyncio.Lock()               # ensures only one switch at a time


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — model loading via Engine Factory
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    PATCH 3 — Hardened lifespan that guarantees any startup failure is
    printed with a full traceback before the process dies.

    The engine is selected by the ASR_ENGINE environment variable, instantiated
    via the factory, and loaded in a ThreadPoolExecutor so the event loop is
    never blocked.
    """
    global asr_engine

    log.info("═" * 62)
    log.info("  Dual-Engine Streaming ASR  —  server starting…")
    log.info("  ASR_ENGINE : %s", ASR_ENGINE_NAME)
    log.info("═" * 62)
    log.info("  torchaudio backend : %s", torchaudio.get_audio_backend())
    log.info("  torch version      : %s", torch.__version__)
    log.info("  CUDA available     : %s", torch.cuda.is_available())

    try:
        # 1. Select the engine via the factory (does NOT load model yet)
        engine = create_engine(ASR_ENGINE_NAME)

        # 2. Load the model in a thread so the event-loop isn't blocked.
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(_EXECUTOR, engine.load_model)

        # 3. Await with a generous timeout to catch hung downloads / deadlocks.
        try:
            await asyncio.wait_for(future, timeout=600)   # 10 min max
        except asyncio.TimeoutError:
            log.exception(
                "CRITICAL STARTUP ERROR — engine.load_model() timed out after 600 s.\n"
                "Check for a hung download or a deadlock in model initialisation."
            )
            raise RuntimeError("Model load timed out") from None

        asr_engine = engine
        _current_engine_key = ASR_ENGINE_NAME

        log.info("  Engine ready  ✓  [%s]", asr_engine.name)
        log.info("═" * 62)

    except Exception:   # noqa: BLE001
        log.exception(
            "CRITICAL STARTUP ERROR — lifespan failed.  "
            "Server will NOT start.  Real error:\n"
        )
        raise

    yield
    log.info("Server shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Dual-Engine Streaming ASR",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def serve_ui():
    return FileResponse("index.html")


@app.get("/health")
async def health():
    """Quick liveness + engine-info endpoint."""
    return {
        "status"       : "ok",
        "engine_loaded": asr_engine is not None,
        "engine_type"  : asr_engine.name if asr_engine else None,
        "cuda"         : torch.cuda.is_available(),
        "device"       : torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "asr_engine_env": ASR_ENGINE_NAME,
    }


@app.get("/api/status")
async def api_status():
    """
    Returns the currently active engine name and switching state.
    Called by the UI to reflect the correct toggle state.
    """
    return {
        "engine_name" : asr_engine.name if asr_engine else None,
        "engine_key"  : _current_engine_key,
        "switching"   : _engine_switching,
        "cuda"        : torch.cuda.is_available(),
        "device"      : torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def _free_engine_memory(engine) -> None:
    """
    Move old engine's model to CPU and release GPU memory.
    Called after a hot-swap to free VRAM from the previous engine.
    """
    try:
        import gc
        model = getattr(engine, '_model', None)
        if model is not None and hasattr(model, 'cpu'):
            model.cpu()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Released GPU memory from old engine: %s", engine.name)
    except Exception as exc:
        log.warning("Could not fully free engine memory (%s): %s", engine.name, exc)


@app.post("/api/switch-engine")
async def api_switch_engine(request: Request):
    """
    Hot-swap the active ASR engine without restarting the server.

    Request body: {"engine": "nemotron" | "qwen"}

    The new engine is loaded in the ThreadPoolExecutor (non-blocking).
    Once ready, the global asr_engine is replaced atomically.
    Old engine GPU memory is freed immediately after swap.
    Existing WebSocket sessions keep using the old engine until they
    disconnect; new sessions will use the new engine.
    """
    global asr_engine, _engine_switching, _current_engine_key

    body = await request.json()
    new_engine_name = body.get("engine", "").strip().lower()

    if not new_engine_name:
        raise HTTPException(status_code=400, detail="'engine' field is required")

    if _engine_switching:
        raise HTTPException(status_code=409, detail="An engine switch is already in progress — please wait")

    # Already on this engine — no-op
    if asr_engine is not None and new_engine_name == _current_engine_key:
        return {"success": True, "engine": asr_engine.name, "engine_key": _current_engine_key,
                "message": "Already using this engine"}

    async with _engine_lock:
        _engine_switching = True
        try:
            log.info("Hot-swap: %s → %s …", _current_engine_key, new_engine_name)

            # 1. Build the new engine object (raises ValueError for unknown names)
            try:
                new_engine = create_engine(new_engine_name)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

            # 2. Load the model in a thread (may take minutes for large models)
            loop = asyncio.get_event_loop()
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(_EXECUTOR, new_engine.load_model),
                    timeout=600,
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504,
                                    detail="Engine load timed out (600 s) — check server logs")

            # 3. Atomic swap
            old_engine = asr_engine
            asr_engine = new_engine
            _current_engine_key = new_engine_name

            # 4. Release old engine VRAM
            if old_engine is not None:
                _free_engine_memory(old_engine)

            log.info("Hot-swap complete — now using %s", asr_engine.name)

        except HTTPException:
            raise
        except Exception as exc:
            log.exception("Hot-swap failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"Engine switch failed: {exc}")
        finally:
            _engine_switching = False

    return {"success": True, "engine": asr_engine.name, "engine_key": _current_engine_key}


# ─────────────────────────────────────────────────────────────────────────────
# Per-Connection Streaming Session
# ─────────────────────────────────────────────────────────────────────────────
# How many samples to accumulate before re-running transcription (0.5 s)
_TRANSCRIBE_EVERY_N_SAMPLES = SAMPLE_RATE // 2   # 8 000 @ 16 kHz
# Maximum audio history kept for context (30 s — Qwen supports up to 30 s)
_MAX_CONTEXT_SAMPLES = SAMPLE_RATE * 30

# Trivial engine outputs to ignore (silence artifacts)
_TRIVIAL_OUTPUTS = {".", "،", "...", "..", ",", "!", "؟", "?"}


class StreamingSession:
    """
    Engine-agnostic streaming ASR session.

    Responsibilities
    ----------------
    - Accumulate incoming PCM chunks.
    - Apply RMS-based VAD to detect silence.
    - Throttle inference calls (every ~0.5 s of new audio).
    - Trigger 'final' events after sustained silence.
    - Delegate all transcription to ``self.engine.run_inference(ctx)``.

    The session holds NO model-specific code — it is fully decoupled from
    NeMo, HuggingFace, or any other ASR backend.
    """

    def __init__(self, engine: BaseASREngine) -> None:
        self.engine           = engine
        self._chunks: list[np.ndarray] = []
        self._pending_samples = 0
        self.silence_frames   = 0
        self.last_partial     = ""
        self.chunk_count      = 0
        self._session_text    = ""  # accumulates all finalized utterances this session
        log.info("StreamingSession init — engine=%s", engine.name)

    # ── Reset ───────────────────────────────────────────────────────────────
    def _reset_for_next_utterance(self) -> None:
        self._chunks          = []
        self._pending_samples = 0
        self.last_partial     = ""
        self.silence_frames   = 0
        self.engine.reset_state()

    # ── Utility ─────────────────────────────────────────────────────────────
    @staticmethod
    def _rms(samples: np.ndarray) -> float:
        return float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0

    # ── Core inference  (runs in ThreadPoolExecutor) ─────────────────────────
    def _run_inference(self, samples: np.ndarray) -> str:
        """
        Accumulate samples, build context window, delegate to engine.

        This method is engine-agnostic:
          1. Append new chunk to buffer.
          2. Wait until 0.5 s of new audio is ready.
          3. Build context window (last ≤ 10 s).
          4. Call engine.run_inference(ctx) → return text.
        """
        self._chunks.append(samples)
        self._pending_samples += len(samples)

        # Wait until 0.5 s of new audio is accumulated
        if self._pending_samples < _TRANSCRIBE_EVERY_N_SAMPLES:
            return self.last_partial

        self._pending_samples = 0

        # Build context window (last ≤ 10 s)
        ctx = np.concatenate(self._chunks)
        if len(ctx) > _MAX_CONTEXT_SAMPLES:
            ctx = ctx[-_MAX_CONTEXT_SAMPLES:]
            self._chunks = [ctx]

        # Delegate to the active ASR engine
        try:
            return self.engine.run_inference(ctx)
        except Exception as exc:
            log.error("Session inference error: %s", exc, exc_info=True)
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

        # Filter trivial engine outputs (silence artifacts like ".")
        if partial.strip() in _TRIVIAL_OUTPUTS or len(partial.strip()) <= 1:
            partial = self.last_partial  # keep showing last valid text

        self.silence_frames = (self.silence_frames + 1) if is_silent else 0

        # Endpointing — sustained silence → emit final
        if self.silence_frames >= SILENCE_FRAMES_TRIGGER and self.last_partial:
            final_text = self.last_partial
            # Accumulate into the session transcript (full session text)
            if final_text:
                self._session_text = (self._session_text + " " + final_text).strip()
            self._reset_for_next_utterance()
            return {"type": "final", "text": self._session_text or final_text}

        # Build full display text: session history + current partial
        if partial:
            full_partial = (self._session_text + " " + partial).strip() if self._session_text else partial
        else:
            full_partial = self._session_text  # nothing new yet

        # Emit partial only when it changes
        if full_partial and full_partial != self.last_partial:
            self.last_partial = full_partial
            return {"type": "partial", "text": full_partial}

        return None

    def flush(self) -> dict | None:
        """Emit any pending partial as final (called on client 'end' signal)."""
        if self.last_partial:
            final = self.last_partial
            self._session_text = ""
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
    session = StreamingSession(asr_engine)

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
