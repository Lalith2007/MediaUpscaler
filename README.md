Here's the **universal, cross-platform README.md** that works for Windows, macOS, and Linux users:

```markdown
# 🎬 MediaUpscaler - AI-Powered Image & Video Enhancement Tool

A professional-grade desktop application that uses **Real-ESRGAN AI** to upscale images and videos to stunning quality (up to 4K), with frame interpolation powered by **RIFE**.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

---

## ✨ Features

- **🖼️ AI Image Upscaling**: Upscale images 2x-4x using state-of-the-art Real-ESRGAN
- **🎥 Video Upscaling**: Enhance videos to 1080p, 2K, or 4K resolution
- **🎞️ Frame Interpolation**: Increase video FPS (30→60, 24→120) using RIFE AI
- **🎨 Color Presets**: Choose from Balanced, Vibrant, Cinematic, or Crisp modes
- **🔐 User Authentication**: Secure login system with SQLite database
- **⚡ GPU Acceleration**: Automatic detection of CUDA (NVIDIA), MPS (Apple Silicon), or CPU
- **🌐 Web Interface**: Modern, responsive UI accessible through any browser
- **🖥️ Cross-Platform**: Works on Windows, macOS, and Linux

---

## 📋 Table of Contents

- [System Requirements](#-system-requirements)
- [Installation](#-installation)
  - [Windows](#-windows)
  - [macOS](#-macos)
  - [Linux](#-linux)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Configuration](#-configuration)
- [Building Standalone App](#-building-standalone-app-optional)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## 💻 System Requirements

### Minimum Requirements
- **OS**: Windows 10+, macOS 11+, or Ubuntu 20.04+ (or equivalent Linux)
- **Python**: 3.9 or higher (3.11+ recommended)
- **RAM**: 8GB (16GB recommended for 4K video)
- **Storage**: 10GB free space
- **GPU**: Optional but highly recommended (NVIDIA with CUDA or Apple Silicon)

### Recommended Requirements
- **RAM**: 16GB or more
- **GPU**: NVIDIA RTX series or Apple M1/M2/M3 chip
- **Storage**: SSD with 20GB+ free space

---

## 🚀 Installation

Choose your operating system and follow the instructions:

---

### 🪟 Windows

#### Step 1: Install Python

1. Download Python from [python.org](https://www.python.org/downloads/)
2. **Important**: During installation, check ✅ **"Add Python to PATH"**
3. Verify installation:
   ```
   python --version
   ```

#### Step 2: Install FFmpeg

**Option A: Using Chocolatey (Recommended)**
```
# Install Chocolatey first (if not installed)
# Run PowerShell as Administrator and paste:
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install FFmpeg
choco install ffmpeg
```

**Option B: Manual Installation**
1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html#build-windows)
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to System PATH:
   - Right-click **This PC** → **Properties** → **Advanced System Settings**
   - Click **Environment Variables**
   - Under **System Variables**, find **Path**, click **Edit**
   - Click **New**, add `C:\ffmpeg\bin`
   - Click **OK** on all windows

4. Verify:
   ```
   ffmpeg -version
   ```

#### Step 3: Clone the Repository

```
# Install Git (if not installed)
# Download from: https://git-scm.com/download/win

# Clone the project
git clone https://github.com/Lalith2007/MediaUpscaler.git
cd MediaUpscaler
```

#### Step 4: Create Virtual Environment

```
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate
```

> ✅ Your prompt should now show `(venv)` at the beginning

#### Step 5: Install Python Dependencies

```
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install flask flask-cors flask-socketio flask-sqlalchemy flask-login werkzeug pillow opencv-python numpy torch torchvision
```

**For NVIDIA GPU (CUDA Support):**
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### Step 6: Download RIFE Binary

```
# Create bin directory
mkdir bin

# Download RIFE for Windows
# Visit: https://github.com/nihui/rife-ncnn-vulkan/releases
# Download: rife-ncnn-vulkan-YYYYMMDD-windows.zip
# Extract rife-ncnn-vulkan.exe to the bin\ folder
```

#### Step 7: Create Required Folders

```
mkdir static\uploads
mkdir static\outputs
mkdir ai_api\weights
```

#### Step 8: Run the Application

```
python ai_api\api_server.py
```

Open browser: **http://localhost:5001**

---

### 🍎 macOS

#### Step 1: Install Homebrew (Package Manager)

```
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Step 2: Install Dependencies

```
# Install Python 3
brew install python@3.11

# Install FFmpeg
brew install ffmpeg

# Install Git
brew install git
```

Verify installations:
```
python3 --version
ffmpeg -version
git --version
```

#### Step 3: Clone the Repository

```
git clone https://github.com/Lalith2007/MediaUpscaler.git
cd MediaUpscaler
```

#### Step 4: Create Virtual Environment

```
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

> ✅ Your prompt should now show `(venv)` at the beginning

#### Step 5: Install Python Dependencies

```
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install flask flask-cors flask-socketio flask-sqlalchemy flask-login werkzeug pillow opencv-python numpy torch torchvision
```

**For Apple Silicon (M1/M2/M3):**
```
# PyTorch with MPS (Metal Performance Shaders) support
pip install torch torchvision
```

#### Step 6: Download RIFE Binary

```
# Create bin directory
mkdir -p bin

# Download RIFE for macOS
curl -L https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-macos.zip -o rife.zip

# Extract
unzip rife.zip

# Move binary
mv rife-ncnn-vulkan-*/rife-ncnn-vulkan bin/rife-video

# Make executable
chmod +x bin/rife-video

# Clean up
rm -rf rife.zip rife-ncnn-vulkan-*
```

#### Step 7: Create Required Folders

```
mkdir -p static/uploads static/outputs ai_api/weights
```

#### Step 8: Run the Application

```
python ai_api/api_server.py
```

Open browser: **http://localhost:5001**

---

### 🐧 Linux (Ubuntu/Debian)

#### Step 1: Install System Dependencies

```
# Update package list
sudo apt update

# Install Python 3 and pip
sudo apt install python3 python3-pip python3-venv -y

# Install FFmpeg
sudo apt install ffmpeg -y

# Install Git
sudo apt install git -y
```

Verify installations:
```
python3 --version
ffmpeg -version
git --version
```

#### Step 2: Clone the Repository

```
git clone https://github.com/Lalith2007/MediaUpscaler.git
cd MediaUpscaler
```

#### Step 3: Create Virtual Environment

```
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

> ✅ Your prompt should now show `(venv)` at the beginning

#### Step 4: Install Python Dependencies

```
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install flask flask-cors flask-socketio flask-sqlalchemy flask-login werkzeug pillow opencv-python numpy torch torchvision
```

**For NVIDIA GPU (CUDA Support):**
```
# First, install NVIDIA drivers and CUDA toolkit
sudo apt install nvidia-driver-525 nvidia-cuda-toolkit -y

# Then install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### Step 5: Download RIFE Binary

```
# Create bin directory
mkdir -p bin

# Download RIFE for Linux
wget https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip

# Extract
unzip rife-ncnn-vulkan-20221029-ubuntu.zip

# Move binary
mv rife-ncnn-vulkan-*/rife-ncnn-vulkan bin/rife-video

# Make executable
chmod +x bin/rife-video

# Clean up
rm -rf rife-ncnn-vulkan-* *.zip
```

#### Step 6: Create Required Folders

```
mkdir -p static/uploads static/outputs ai_api/weights
```

#### Step 7: Run the Application

```
python ai_api/api_server.py
```

Open browser: **http://localhost:5001**

---

## 🎯 Quick Start

Once the server is running, follow these steps:

1. **Register an Account**
   - Click **Register** on the homepage
   - Create username and password
   - You'll be automatically logged in

2. **Test Image Upscaling**
   - Click **Image Upscaler**
   - Upload a test image
   - Select **Balanced** preset
   - Click **Upscale Image**
   - Wait for processing (30-60 seconds)
   - Download the result

3. **You're Ready!**
   - The AI model downloads automatically on first use (~67MB)
   - Subsequent upscales will be faster

---

## 📖 Usage Guide

### Image Upscaling

1. Navigate to **Image Upscaler**
2. Click **Upload Image** (supports JPG, PNG, JPEG)
3. Select a **Color Preset**:
   - **Balanced**: Natural, realistic enhancement (recommended)
   - **Vibrant**: Boosted colors and contrast
   - **Cinematic**: Film-like color grading with muted tones
   - **Crisp**: Maximum sharpness and detail
4. Click **Upscale Image**
5. Progress bar shows AI processing status
6. Click **Download** to save your enhanced image

**Tips:**
- Larger images take longer to process
- GPU acceleration significantly speeds up processing
- Try different presets to see what looks best

### Video Upscaling

1. Navigate to **Video Upscaler**
2. Upload video (MP4, MOV, AVI supported)
3. Select **Target Resolution**:
   - **1080p** (1920×1080) - Standard HD
   - **2K** (2560×1440) - High quality
   - **4K** (3840×2160) - Ultra HD
4. Select a **Color Preset**
5. Click **Upscale Video**
6. Wait for processing (can take 5-30 minutes depending on video length)
7. Download the enhanced video

**Tips:**
- Start with shorter videos (<1 minute) to test
- Video upscaling is very resource-intensive
- GPU acceleration is highly recommended
- Original audio is preserved

### Frame Interpolation

1. Navigate to **Frame Interpolator**
2. Upload a video
3. Select **Target FPS**:
   - **30 FPS** - Smooth motion
   - **60 FPS** - Very smooth (recommended)
   - **120 FPS** - Ultra smooth (experimental)
   - **240 FPS** - Extreme slow motion
4. Click **Interpolate**
5. Download the smoothed video

**Tips:**
- Best for low-FPS videos (24fps, 30fps)
- Creates new frames between existing ones
- Great for making old footage look modern

---

## ⚙️ Configuration

### Changing Server Port

Edit `ai_api/api_server.py` (bottom of file):

```
socketio.run(app, host='localhost', port=5001, debug=False)
```

Change `5001` to any available port.

### GPU Configuration

The app automatically detects your GPU:
- **NVIDIA CUDA** (Windows/Linux)
- **Apple Metal (MPS)** (macOS M1/M2/M3)
- **CPU Fallback** (all systems)

**To force CPU mode:**

**Windows:**
```
set CUDA_VISIBLE_DEVICES=
python ai_api\api_server.py
```

**macOS/Linux:**
```
export CUDA_VISIBLE_DEVICES=""
python ai_api/api_server.py
```

### Database Location

User accounts are stored in:
- **Windows**: `C:\Users\YourName\mediaupscaler_users.db`
- **macOS/Linux**: `~/mediaupscaler_users.db`

---

## 📂 Project Structure

```
MediaUpscaler/
├── ai_api/
│   ├── api_server.py          # Main Flask application
│   └── weights/               # AI model weights (auto-downloaded)
├── templates/                 # HTML templates
│   ├── index.html             # Landing page
│   ├── login.html             # Login page
│   ├── register.html          # Registration page
│   ├── dashboard.html         # Main dashboard
│   ├── image-upscaler.html    # Image upscaler interface
│   ├── video-upscaler.html    # Video upscaler interface
│   ├── frame-interpolator.html# Frame interpolation interface
│   └── base.html              # Base template
├── static/
│   ├── uploads/               # Temporary user uploads (auto-created)
│   └── outputs/               # Processed files (auto-created)
├── bin/
│   └── rife-video             # Frame interpolation binary
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
└── requirements.txt           # Python dependencies (optional)
```

---

## 🏗️ Building Standalone App (Optional)

You can package MediaUpscaler as a standalone executable:

### Install PyInstaller

```
pip install pyinstaller
```

### Build the App

**Windows:**
```
pyinstaller --noconfirm --onedir --windowed --clean ^
  --name "MediaUpscaler" ^
  --add-data "templates;templates" ^
  --add-data "static;static" ^
  --add-data "bin;bin" ^
  --hidden-import "engineio.async_drivers.threading" ^
  --hidden-import "flask_sqlalchemy" ^
  --hidden-import "flask_login" ^
  --exclude-module "torch.utils.tensorboard" ^
  ai_api\api_server.py
```

**macOS/Linux:**
```
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

The executable will be in `dist/MediaUpscaler/`

---

## 🐛 Troubleshooting

### "Address already in use" Error

**Windows:**
```
netstat -ano | findstr :5001
taskkill /PID <PID> /F
```

**macOS/Linux:**
```
lsof -ti:5001 | xargs kill -9
```

### "Module not found" Error

Make sure virtual environment is activated:
```
# Check if (venv) appears in your prompt
# If not, activate it:

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### "FFmpeg not found" Error

Verify FFmpeg is installed and in PATH:
```
ffmpeg -version
```

If not found, reinstall FFmpeg using the instructions for your OS.

### "CUDA out of memory" Error

Your GPU doesn't have enough VRAM. Solutions:
1. Process smaller images/videos
2. Use CPU mode (see Configuration section)
3. Close other GPU-intensive applications

### Video Processing is Very Slow

- **Enable GPU acceleration** (see Configuration)
- **Process shorter videos** (1-2 minutes max)
- **Lower target resolution** (try 1080p instead of 4K)
- **Check GPU usage**: Use Task Manager (Windows) or Activity Monitor (macOS)

### Database Locked Error

Delete and recreate the database:

**Windows:**
```
del %USERPROFILE%\mediaupscaler_users.db
```

**macOS/Linux:**
```
rm ~/mediaupscaler_users.db
```

Restart the app and register again.

### "RIFE binary not found" Error

Make sure the binary is executable:

**macOS/Linux:**
```
chmod +x bin/rife-video
```

**Windows:** No action needed (`.exe` files are executable by default)

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** a feature branch:
   ```
   git checkout -b feature/AmazingFeature
   ```
3. **Commit** your changes:
   ```
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push** to the branch:
   ```
   git push origin feature/AmazingFeature
   ```
5. **Open** a Pull Request

### Development Setup

```
# Clone your fork
git clone https://github.com/YourUsername/MediaUpscaler.git
cd MediaUpscaler

# Create branch
git checkout -b feature/my-feature

# Make changes, test, commit
git add .
git commit -m "Description of changes"
git push origin feature/my-feature
```

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Real-ESRGAN** by Xintao Wang ([GitHub](https://github.com/xinntao/Real-ESRGAN))
- **RIFE** by Zhewei Huang ([GitHub](https://github.com/megvii-research/ECCV2022-RIFE))
- **Flask** by Pallets Projects ([Website](https://flask.palletsprojects.com/))
- **PyTorch** by Meta AI ([Website](https://pytorch.org/))

---

## 📞 Support

### Getting Help

1. **Check the [Troubleshooting](#-troubleshooting) section**
2. **Search [existing issues](https://github.com/Lalith2007/MediaUpscaler/issues)**
3. **Open a [new issue](https://github.com/Lalith2007/MediaUpscaler/issues/new)** with:
   - Operating system and version
   - Python version (`python --version`)
   - Full error message (copy from terminal)
   - Steps to reproduce the problem

### Useful Commands

**Check Python version:**
```
python --version  # or python3 --version
```

**Check if GPU is detected:**
```
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

**Test FFmpeg:**
```
ffmpeg -version
```

---

## ⭐ Star This Repo

If this project helped you, please give it a ⭐ on GitHub!

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

**Made with ❤️ by [Lalith](https://github.com/Lalith2007)**
```


