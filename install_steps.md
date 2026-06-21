# 🚀 Ara-Nemotron Streaming ASR — دليل النشر الكامل على السيرفر
> موديل: `Abdelkareem/Ara-nemotron-3.5-asr-streaming-0.6b`  
> Stack: FastAPI · NeMo · CUDA · RTX 3080  
> آخر تحديث: 2026-06-17 ✅ مُختبر ويشتغل

---

## ⚠️ تحذيرات مهمة قبل أي حاجة

| المتطلب | الإصدار الصح | سبب الـ Pin |
|---|---|---|
| Python | **3.11 64-bit فقط** | NeMo مش شغال على 3.12/3.13 |
| torch | **2.5.1+cu121** | nemo-toolkit 2.7.x محتاجه |
| pyarrow | **17.x (مش 24.x)** | 24.x بيعمل Windows Access Violation |
| datasets | **2.x (مش 5.x)** | 5.x محتاج pyarrow >= 21 |
| numpy | **< 2.0** | NeMo C-extensions بتكسر مع numpy 2.x |

---

## 📦 الخطوة 1 — تجهيز السيرفر

### 1.1 المتطلبات الأساسية على السيرفر
```bash
# تأكد إن CUDA drivers مسطبة (للـ GPU)
nvidia-smi   # لازم يشتغل ويظهر GPU

# تأكد من Python 3.11
python3.11 --version   # Linux
py -3.11 --version     # Windows
```

### 1.2 رفع ملفات المشروع على السيرفر
انسخ المجلد بالكامل على السيرفر. المحتوى المطلوب:
```
your-project/
├── app.py              ← الملف الرئيسي (النسخة المُصلحة)
├── index.html          ← واجهة المستخدم
├── requirements.txt    ← الـ dependencies المُثبتة
├── install_steps.md    ← هذا الملف
└── model/
    └── model.nemo      ← ملف الموديل (مهم جداً!)
```

**رفع الملفات بـ SCP (من جهازك للسيرفر):**
```bash
# من جهازك الـ Windows — PowerShell
scp -r "d:\test\trails\fine tunning\*" user@SERVER_IP:/home/user/asr-server/

# أو لو عندك rsync
rsync -avz --progress "d:/test/trails/fine tunning/" user@SERVER_IP:/home/user/asr-server/
```

---

## 🐍 الخطوة 2 — إعداد بيئة Python على السيرفر

### 2.1 إنشاء البيئة الافتراضية
```bash
cd /home/user/asr-server

# إنشاء venv بـ Python 3.11
python3.11 -m venv .venv

# تفعيل البيئة (Linux/Mac)
source .venv/bin/activate

# تفعيل البيئة (Windows Server)
.venv\Scripts\activate
```

### 2.2 تسطيب PyTorch مع CUDA أولاً
```bash
# ⚠️ لازم يتسطب لوحده الأول قبل باقي المكاتب
# CUDA 12.1 (للـ RTX 3080 / RTX 3090 / A100)
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# تحقق إن CUDA شغالة
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
```

> **لو السيرفر مش عنده GPU** استخدم:
> ```bash
> pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
> ```

### 2.3 تسطيب باقي المكاتب
```bash
# تسطيب كل حاجة من requirements.txt
pip install -r requirements.txt --no-cache-dir

# ✅ الـ output المتوقع (ابحث عن هذا في الآخر)
# Successfully installed datasets-2.21.0 pyarrow-17.0.0 ...
```

### 2.4 إنشاء مجلد Temp على الدرايف الكبير
```bash
# Linux — NeMo بيفك .nemo files في /tmp (تأكد فيه مساحة كافية ~5GB)
df -h /tmp   # لازم يكون فيه مساحة

# لو /tmp مش كافية، غيّر مسار الـ temp في app.py
# ابحث عن _NEMO_TMPDIR في app.py وغيّره لمجلد عنده مساحة
# مثال على Linux:
# _NEMO_TMPDIR = "/data/tmp"   # أو أي partition كبير
```

---

## ▶️ الخطوة 3 — تشغيل السيرفر

### 3.1 تشغيل للتجربة (Development)
```bash
# تأكد إنك جوه البيئة (source .venv/bin/activate)
python app.py

# أو مباشرة بـ uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**الـ Output المتوقع لما يشتغل صح:**
```
INFO:     Started server process [XXXX]
INFO:     Waiting for application startup.
══════════════════════════════════════════════
  Ara-Nemotron Streaming ASR  —  server starting…
══════════════════════════════════════════════
  torchaudio backend : soundfile
  torch version      : 2.5.1+cu121
  CUDA available     : True
  Loading .nemo from dir: model/model.nemo
  Loaded as EncDecHybridRNNTCTCBPEModelWithPrompt (strict=False) ✔
  GPU  : NVIDIA GeForce RTX 3080 (16.0 GB VRAM)
  Model ready  ✓
══════════════════════════════════════════════
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3.2 تشغيل للإنتاج (Production) — بـ systemd على Linux
```bash
# إنشاء service file
sudo nano /etc/systemd/system/asr-server.service
```

```ini
[Unit]
Description=Ara-Nemotron ASR Streaming Server
After=network.target

[Service]
Type=simple
User=user
WorkingDirectory=/home/user/asr-server
Environment="PATH=/home/user/asr-server/.venv/bin"
Environment="TMPDIR=/data/tmp"
ExecStart=/home/user/asr-server/.venv/bin/python app.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# تفعيل وتشغيل السيرفس
sudo systemctl daemon-reload
sudo systemctl enable asr-server
sudo systemctl start asr-server
sudo systemctl status asr-server

# متابعة الـ logs
sudo journalctl -u asr-server -f
```

---

## 🧪 الخطوة 4 — اختبار السيرفر (الـ API Endpoints)

### Endpoint 1: Health Check
**اتحقق إن السيرفر والموديل شغالين:**
```bash
curl http://SERVER_IP:8000/health
```
**الـ Response المتوقع:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "cuda": true,
  "device": "NVIDIA GeForce RTX 3080 Laptop GPU",
  "model_id": "Abdelkareem/Ara-nemotron-3.5-asr-streaming-0.6b"
}
```
> ✅ لو `model_loaded: true` — السيرفر جاهز بالكامل

---

### Endpoint 2: واجهة المستخدم (Web UI)
افتح في المتصفح:
```
http://SERVER_IP:8000/
```
- ✅ هتلاقي واجهة **Dark Mode** فيها زر "Start Recording"
- اضغط الزر، تكلم بالعربي، وهيطلعلك النص في الـ Real-Time

---

### Endpoint 3: WebSocket Transcription
**للتكامل مع تطبيقك الخاص:**
```
ws://SERVER_IP:8000/ws/transcribe
```

**بروتوكول الاتصال:**
```javascript
// 1. افتح الـ WebSocket
const ws = new WebSocket("ws://SERVER_IP:8000/ws/transcribe");

// 2. ابعت Audio chunks (PCM int16, mono, 16kHz)
ws.send(pcmAudioBytes);  // Binary data

// 3. استقبل النتائج
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === "partial") {
    // نص مؤقت أثناء الكلام
    console.log("جاري التعرف:", data.text, "| زمن:", data.latency_ms, "ms");
  }
  
  if (data.type === "final") {
    // نص نهائي بعد لحظة صمت
    console.log("النص النهائي:", data.text);
  }
  
  if (data.type === "eos") {
    // نهاية الجلسة
    ws.close();
  }
};

// 4. لما تخلص التسجيل، ابعت إشارة الإنهاء
ws.send(JSON.stringify({ type: "end" }));

// 5. Ping/Pong للـ keepalive
ws.send(JSON.stringify({ type: "ping" }));
// هيرجعلك: { type: "pong" }
```

**مواصفات الـ Audio المطلوبة:**
| الخاصية | القيمة |
|---|---|
| Format | PCM Raw (int16-LE) |
| Sample Rate | **16,000 Hz** |
| Channels | **Mono (1 channel)** |
| Bit Depth | 16-bit |

---

### Endpoint 4: اختبار بـ Python Script
```python
import asyncio
import websockets
import wave

async def test_transcription(audio_file: str, server: str = "localhost:8000"):
    """اختبر الـ transcription بملف WAV."""
    uri = f"ws://{server}/ws/transcribe"
    
    async with websockets.connect(uri) as ws:
        # افتح ملف WAV
        with wave.open(audio_file, 'rb') as wf:
            assert wf.getnchannels() == 1, "لازم Mono"
            assert wf.getframerate() == 16000, "لازم 16kHz"
            
            chunk_size = 16000 // 4  # 250ms chunks
            while chunk := wf.readframes(chunk_size):
                await ws.send(chunk)
                
                # استقبل أي رد جاهز
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    data = json.loads(msg)
                    print(f"[{data['type']}] {data.get('text', '')}")
                except asyncio.TimeoutError:
                    pass
        
        # أنهي الجلسة
        await ws.send(json.dumps({"type": "end"}))
        
        # استقبل الـ final
        async for msg in ws:
            data = json.loads(msg)
            print(f"[{data['type']}] {data.get('text', '')}")
            if data['type'] == 'eos':
                break

# تشغيل الاختبار
asyncio.run(test_transcription("test_arabic.wav", "SERVER_IP:8000"))
```

---

## 🔐 الخطوة 5 — Nginx Reverse Proxy (للإنتاج)

```nginx
# /etc/nginx/sites-available/asr-server
server {
    listen 80;
    server_name your-domain.com;

    # HTTP → HTTPS redirect (لو عندك SSL)
    # return 301 https://$host$request_uri;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        
        # ⚠️ مهم جداً للـ WebSocket
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # Timeout طويل عشان الـ WebSocket مش بيخلص بسرعة
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/asr-server /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

---

## 🐛 مشاكل شائعة وحلولها

| المشكلة | السبب | الحل |
|---|---|---|
| `Windows fatal exception: access violation` | pyarrow >= 18 | `pip install "pyarrow>=17,<18"` |
| `AttributeError: module 'pyarrow' has no attribute 'json_'` | pyarrow قديم + datasets جديد | `pip install "datasets>=2.18,<3"` |
| `OSError: [Errno 28] No space left on device` | الـ /tmp ممتلية | غيّر `_NEMO_TMPDIR` في app.py لمكان فيه مساحة |
| `CUDA unavailable` | Driver مش مسطب أو مش متوافق | `nvidia-smi` → تأكد من CUDA version |
| `Application startup failed` (بدون error) | Exception متبلعة | الكود الحالي بيطبع الـ error دايماً |
| السيرفر بيقفل في صمت | نفس المشكلة القديمة | `faulthandler` مفعّل في app.py — هيطبع الـ crash |

---

## ✅ Checklist قبل التشغيل

```
[ ] Python 3.11 مسطب
[ ] torch 2.5.1+cu121 مسطب (nvidia-smi شغال)
[ ] pyarrow 17.x مسطب (مش 24.x)
[ ] datasets 2.x مسطب (مش 5.x)
[ ] numpy < 2.0 مسطب
[ ] ملف model/model.nemo موجود
[ ] D:\tmp (أو /data/tmp) موجود وفيه مساحة > 3GB
[ ] python app.py شغال محلياً بدون error
[ ] http://localhost:8000/health يرجع model_loaded: true
```