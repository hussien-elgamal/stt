# Ara-Nemotron Real-time Streaming ASR

A local, end-to-end streaming ASR prototype built with **FastAPI + NVIDIA NeMo** and a **Vanilla JS / dark-mode** frontend.

| Item | Detail |
|------|--------|
| Model | `Abdelkareem/Ara-nemotron-3.5-asr-streaming-0.6b` |
| Architecture | FastConformer-RNNT (cache-aware streaming) |
| Language | Arabic · Egyptian Arabic |
| GPU | NVIDIA RTX 3080 (10 GB VRAM) |
| Audio in | 16 kHz · mono · PCM Int16 |
| Chunk size | 250 ms |

---

## Project structure

```
fine tunning/
├── app.py            ← FastAPI backend (WebSocket + NeMo)
├── index.html        ← Frontend UI (Vanilla JS, dark mode)
├── requirements.txt  ← Python dependencies
├── README.md         ← This file
└── model/            ← Place your .nemo file here  (optional)
```

---

## 1 · Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10 – 3.11 |
| CUDA Toolkit | 12.x (matching your driver) |
| PyTorch | 2.2+ (CUDA build) |
| NVIDIA NeMo | 2.0+ |

> **Windows users**: NeMo runs on WSL2 or a native Python env. A **WSL2 Ubuntu 22.04** setup is strongly recommended.

---

## 2 · Installation

### Step 1 — Create a virtual environment

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux / WSL2
source .venv/bin/activate
```

### Step 2 — Install PyTorch (CUDA 12.1 build)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> Verify CUDA: `python -c "import torch; print(torch.cuda.get_device_name(0))"`

### Step 3 — Install NeMo and server deps

```bash
pip install -r requirements.txt
```

NeMo is a large package (~2 GB). This step takes several minutes on first run.

---

## 3 · Model setup (manual download)

Since you're downloading the model manually, save the `.nemo` file (or the full HuggingFace repo) into the `model/` directory:

```
fine tunning/
└── model/
    └── Ara-nemotron-3.5-asr-streaming-0.6b.nemo
```

The server checks **`./model/`** on startup and auto-discovers any `.nemo` file inside.

### Download via huggingface-cli (alternative)

```bash
pip install huggingface_hub
huggingface-cli download Abdelkareem/Ara-nemotron-3.5-asr-streaming-0.6b \
  --local-dir ./model
```

Then find the `.nemo` file in the downloaded directory and move/rename as needed.

### Override path at runtime

```bash
MODEL_PATH=C:\path\to\my\model.nemo uvicorn app:app
```

---

## 4 · Running the server

```bash
# From the project directory
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Expected startup output:**
```
════════════════════════════════════════════════════════════
  Ara-Nemotron Streaming ASR  —  server starting…
════════════════════════════════════════════════════════════
  Loading .nemo from dir: model/Ara-nemotron-3.5-...nemo
  GPU  : NVIDIA GeForce RTX 3080 (10.0 GB VRAM)
  Model ready  ✓
════════════════════════════════════════════════════════════
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Open **http://localhost:8000** in Chrome or Edge.

---

## 5 · Using the UI

1. Click **Start Recording** — browser will ask for microphone permission.
2. Speak in Arabic (Modern Standard or Egyptian dialect).
3. The **Live Partial** panel (right-to-left) updates in real time as you speak.
4. When you pause for ~1.5 seconds the partial is committed as a **Final** segment and appended to the log.
5. Click **Stop Recording** to end the session; any remaining partial is flushed.

---

## 6 · Architecture

```
Browser                          Server (FastAPI)
──────────────────────────────   ──────────────────────────────────────
Microphone
  ↓ getUserMedia
AudioContext (16 kHz)
  ↓ ScriptProcessorNode
Accumulate Float32 samples
  ↓ every 250 ms
downsample → Int16 PCM           ← WebSocket binary frame
                                   ↓
                                 np.frombuffer → float32
                                   ↓
                                 CacheAwareStreamingAudioBuffer.append()
                                   ↓
                                 model.transcribe(batch)   [GPU]
                                   ↓
                                 {"type":"partial","text":"…"}  →  WS
                                 (or "final" after silence)
```

### Key latency contributors

| Stage | Typical cost |
|-------|-------------|
| Chunk accumulation | 250 ms (by design) |
| Network (localhost) | < 1 ms |
| GPU inference (RTX 3080) | 30 – 150 ms |
| Total round-trip | **~300 – 450 ms** |

---

## 7 · Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./model` | Path to `.nemo` file or directory |
| `SILENCE_RMS` | `0.007` | RMS energy threshold for silence detection |
| `SILENCE_FRAMES` | `6` | Consecutive silent chunks before a "final" is emitted (~1.5 s) |

Example:
```bash
MODEL_PATH=./model SILENCE_RMS=0.005 SILENCE_FRAMES=8 uvicorn app:app --port 8000
```

---

## 8 · Troubleshooting

| Symptom | Fix |
|---------|-----|
| `CUDA not available` | Reinstall PyTorch with CUDA build; run `nvidia-smi` to verify driver |
| `ModuleNotFoundError: nemo` | Run `pip install nemo_toolkit[asr]` inside your venv |
| No transcript output | Check RMS threshold (`SILENCE_RMS`); speak closer to mic |
| Very high latency | Reduce `CHUNK_MS` in `index.html` to 100–160 ms |
| Browser blocks mic | Serve over HTTPS or use `localhost` (which is allowed by browsers) |
| `WebSocket error` | Confirm server is running and port 8000 is accessible |

---

## 9 · Tuning tips

- **Chunk size**: Smaller chunks (100–160 ms) reduce display latency but increase server call frequency. The NeMo model was trained on 80–320 ms chunks, so stay in that range.
- **Silence threshold**: Lower `SILENCE_RMS` if the endpoint triggers too eagerly in noisy environments; raise it for very quiet speakers.
- **Silence frames**: Increase `SILENCE_FRAMES` for longer pauses before committing a final (e.g., `10` ≈ 2.5 s).
- **GPU memory**: The 0.6B model fits comfortably in ~2 GB of VRAM, leaving plenty of headroom on an RTX 3080.
