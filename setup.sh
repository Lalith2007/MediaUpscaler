#!/usr/bin/env bash
# ============================================================
# MediaUpscaler — One-shot Setup Script (macOS)
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "🎬 MediaUpscaler — Setup"
echo "========================"

# ---- 1. Virtual Environment ----
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment (with system-site-packages for PyTorch)…"
    python3 -m venv --system-site-packages venv
else
    echo "✅ Virtual environment already exists."
fi

source venv/bin/activate
echo "   → Using Python: $(python --version) at $(which python)"

# ---- 2. Install Python dependencies ----
echo "📦 Installing Python dependencies…"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "✅ Python dependencies installed."

# ---- 3. Verify torch is available (via system-site-packages) ----
echo "🔍 Verifying PyTorch availability…"
python -c "import torch; print(f'   → PyTorch {torch.__version__}  |  MPS: {torch.backends.mps.is_available()}  |  CUDA: {torch.cuda.is_available()}')"

# ---- 4. Create required directories ----
echo "📁 Creating required directories…"
mkdir -p static/uploads static/outputs ai_api/weights bin
echo "✅ Directories ready."

# ---- 5. Download RIFE binary for macOS (if missing) ----
if [ ! -f "bin/rife-video" ]; then
    echo "⬇️  Downloading RIFE (rife-ncnn-vulkan) for macOS…"
    RIFE_URL="https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-macos.zip"
    curl -L "$RIFE_URL" -o /tmp/rife_mac.zip --progress-bar
    unzip -o /tmp/rife_mac.zip -d /tmp/rife_extract > /dev/null 2>&1
    # Find the binary inside the extracted folder
    RIFE_BIN=$(find /tmp/rife_extract -name "rife-ncnn-vulkan" -type f | head -1)
    if [ -n "$RIFE_BIN" ]; then
        cp "$RIFE_BIN" bin/rife-video
        chmod +x bin/rife-video
        echo "✅ RIFE binary installed to bin/rife-video"
    else
        echo "⚠️  Could not locate rife-ncnn-vulkan binary in the archive. Frame interpolation may not work."
    fi
    rm -rf /tmp/rife_mac.zip /tmp/rife_extract
else
    echo "✅ RIFE binary already present."
fi

echo ""
echo "🎉 Setup complete!"
echo "   Run the app with:  ./run.sh"
echo ""
