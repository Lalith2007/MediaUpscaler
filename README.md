# 🎬 MediaUpscaler — AI-Powered Image & Video Enhancement

A professional-grade web application that uses **Real-ESRGAN AI** to upscale images and videos to stunning quality (up to 4K), with frame interpolation powered by **RIFE**.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

---

## ✨ Features

- 🖼️ **AI Image Upscaling** — Upscale images 2x–4x using state-of-the-art Real-ESRGAN
- 🎥 **Video Upscaling** — Enhance videos to 1080p, 2K, or 4K resolution
- 🎞️ **Frame Interpolation** — Increase video FPS (30→60, 24→120) using RIFE AI
- 🎨 **Color Presets** — Choose from Balanced, Vibrant, Cinematic, or Crisp modes
- 🔐 **User Authentication** — Google OAuth login + local developer bypass
- ⚡ **GPU Acceleration** — Automatic detection of CUDA (NVIDIA), MPS (Apple Silicon), or CPU
- 🌐 **Web Interface** — Modern, responsive UI accessible through any browser
- 🖥️ **Cross-Platform** — Works on Windows, macOS, and Linux

---

## 💻 System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| **OS** | Windows 10+, macOS 11+, Ubuntu 20.04+ | — |
| **Python** | 3.9+ | 3.11+ |
| **RAM** | 8 GB | 16 GB (for 4K video) |
| **Storage** | 10 GB free | SSD with 20 GB+ free |
| **GPU** | Optional (CPU works) | NVIDIA RTX / Apple M1–M3 |

---

## 🚀 Quick Start (macOS / Linux)

### 1. Clone the repository
```bash
git clone https://github.com/Lalith2007/MediaUpscaler.git
cd MediaUpscaler
```

### 2. Run the automated setup
```bash
chmod +x setup.sh run.sh
./setup.sh
```

This will automatically:
- Create a Python virtual environment
- Install all dependencies
- Download the RIFE binary for frame interpolation
- Create required folders (`static/uploads`, `static/outputs`, `ai_api/weights`)

### 3. (Optional) Configure Google Login
Create a `.env` file in the project root:
```env
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
```
> **Note:** If you skip this step, you can still use the app via the **⚡ Bypass (Local Dev)** button on the homepage. No credentials required for local use.

### 4. Start the application
```bash
./run.sh
```

This starts both servers:
- 🌐 **Web App** → [http://localhost:8000](http://localhost:8000)
- 🧠 **AI API** → [http://localhost:5001](http://localhost:5001)

Press **Ctrl+C** to stop both servers.

---

## 🪟 Quick Start (Windows)

### 1. Prerequisites

- **Python 3.9+**: Download from [python.org](https://www.python.org/downloads/). Check ✅ **"Add Python to PATH"** during installation.
- **FFmpeg**: Install via [Chocolatey](https://chocolatey.org/) (`choco install ffmpeg`) or download from [ffmpeg.org](https://ffmpeg.org/download.html#build-windows) and add to PATH.
- **Git**: Download from [git-scm.com](https://git-scm.com/download/win).

### 2. Clone and set up
```bash
git clone https://github.com/Lalith2007/MediaUpscaler.git
cd MediaUpscaler

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download RIFE binary
```bash
mkdir bin
# Download from: https://github.com/nihui/rife-ncnn-vulkan/releases
# Extract rife-ncnn-vulkan.exe into the bin\ folder
```

### 4. Create required folders
```bash
mkdir static\uploads
mkdir static\outputs
mkdir ai_api\weights
```

### 5. (Optional) Configure Google Login
Create a `.env` file in the project root with your Google OAuth credentials (see macOS/Linux section above).

### 6. Start the application
Open **two separate terminals**:

**Terminal 1 — AI Server:**
```bash
venv\Scripts\activate
python ai_api\api_server.py
```

**Terminal 2 — Web App:**
```bash
venv\Scripts\activate
python app.py
```

Open browser → [http://localhost:8000](http://localhost:8000)

---

## 📖 Usage Guide

### Image Upscaling
1. Navigate to **Image Upscaler**
2. Upload an image (JPG, PNG, JPEG)
3. Select a **Color Preset**: Balanced, Vibrant, Cinematic, or Crisp
4. Click **Start Upscaling**
5. Download the enhanced result

> **Tip:** The AI model weights (~67 MB) download automatically on first use.

### Video Upscaling
1. Navigate to **Video Upscaler**
2. Upload a video (MP4, MOV, AVI)
3. Select target resolution: **1080p**, **2K**, or **4K**
4. Select a color preset
5. Click **Upscale Video** and wait for processing
6. Download the enhanced video (original audio is preserved)

> **Tip:** Start with short videos (< 1 minute). GPU acceleration is highly recommended.

### Frame Interpolation
1. Navigate to **Frame Interpolator**
2. Upload a video
3. Select target FPS: 30, 60, 120, or 240
4. Click **Interpolate** and download the smoothed video

> **Tip:** Best for low-FPS footage (24fps, 30fps). Great for making old footage look modern.

---

## ⚙️ Configuration

### Changing Server Port
Edit the bottom of `app.py` (web app port) or `ai_api/api_server.py` (AI server port):
```python
socketio.run(app, host='localhost', port=8000, ...)   # Web App
socketio.run(app, host='localhost', port=5001, ...)   # AI API
```

### GPU Configuration
The app automatically detects your GPU:
- **NVIDIA CUDA** (Windows/Linux)
- **Apple Metal (MPS)** (macOS M1/M2/M3)
- **CPU Fallback** (all systems)

To force CPU mode:
```bash
export CUDA_VISIBLE_DEVICES=""    # macOS/Linux
set CUDA_VISIBLE_DEVICES=         # Windows
```

### Check GPU Detection
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

---

## 📂 Project Structure

```
MediaUpscaler/
├── ai_api/
│   ├── api_server.py          # AI inference server (Real-ESRGAN + RIFE)
│   └── weights/               # AI model weights (auto-downloaded)
├── templates/                 # HTML templates (Jinja2)
│   ├── base.html              # Base layout
│   ├── index.html             # Landing page
│   ├── dashboard.html         # Main dashboard
│   ├── image-upscaler.html    # Image upscaler interface
│   ├── video-upscaler.html    # Video upscaler interface
│   └── frame-interpolator.html# Frame interpolation interface
├── static/
│   ├── uploads/               # Temporary user uploads
│   └── outputs/               # Processed output files
├── bin/
│   └── rife-video             # Frame interpolation binary
├── app.py                     # Flask web application
├── setup.sh                   # Automated setup script (macOS/Linux)
├── run.sh                     # Unified server launcher (macOS/Linux)
├── requirements.txt           # Python dependencies
├── .env                       # Google OAuth credentials (optional)
└── README.md                  # This file
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| **"Address already in use"** | `lsof -ti:5001,8000 \| xargs kill -9` (macOS/Linux) or `netstat -ano \| findstr :5001` + `taskkill /PID <PID> /F` (Windows) |
| **"Module not found"** | Activate venv first: `source venv/bin/activate` (macOS/Linux) or `venv\Scripts\activate` (Windows) |
| **"FFmpeg not found"** | Install FFmpeg: `brew install ffmpeg` (macOS) or `choco install ffmpeg` (Windows) |
| **"CUDA out of memory"** | Use smaller images/videos, close GPU-intensive apps, or force CPU mode |
| **Video processing is slow** | Enable GPU, process shorter videos, or lower target resolution |
| **Database locked** | Delete `instance/editing_suite.db` and restart the app |
| **"RIFE binary not found"** | Run `chmod +x bin/rife-video` (macOS/Linux) |

---

## 🏗️ Building Standalone App (Optional)

```bash
pip install pyinstaller

# macOS/Linux
pyinstaller --noconfirm --onedir --windowed --clean \
  --name "MediaUpscaler" \
  --add-data "templates:templates" \
  --add-data "static:static" \
  --add-data "bin:bin" \
  --hidden-import "engineio.async_drivers.threading" \
  --hidden-import "flask_sqlalchemy" \
  --hidden-import "flask_login" \
  --exclude-module "torch.utils.tensorboard" \
  ai_api/api_server.py
```

The executable will be in `dist/MediaUpscaler/`.

---

## 🙏 Acknowledgments

- **[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)** by Xintao Wang
- **[RIFE](https://github.com/megvii-research/ECCV2022-RIFE)** by Zhewei Huang
- **[Flask](https://flask.palletsprojects.com/)** by Pallets Projects
- **[PyTorch](https://pytorch.org/)** by Meta AI

---

## 📈 Roadmap

- [ ] Batch processing (multiple images/videos)
- [ ] Drag-and-drop file upload
- [ ] Real-time preview
- [ ] Custom color grading profiles
- [ ] Docker support
- [ ] Web API for developers
- [ ] Mobile app (iOS/Android)

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

**Made with ❤️ by [Lalith](https://github.com/Lalith2007)**
