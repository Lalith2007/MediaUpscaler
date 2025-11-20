from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io
import base64
import cv2
import time
import os
import tempfile
import shutil
import subprocess
import sys
import math


# ===== PYINSTALLER PATH HELPER =====
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# ===== FLASK SETUP =====
# We update Flask to look for templates/static in the correct resource path
app = Flask(__name__,
            template_folder=resource_path('templates'),
            static_folder=resource_path('static'))
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# ===== CONFIGURATION =====
AI_AVAILABLE = False
model = None
device = None
current_job_id = None

# Update paths to use resource_path so they work inside the App bundle
MODEL_DIR = resource_path('train_log')
WEIGHTS_PATH = resource_path(os.path.join('weights', 'RealESRGAN_x4plus.pth'))

# ===== 1. EMBEDDED REAL-ESRGAN ARCHITECTURE =====
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    def make_layer(basic_block, num_basic_block, **kwarg):
        layers = []
        for _ in range(num_basic_block):
            layers.append(basic_block(**kwarg))
        return nn.Sequential(*layers)


    class ResidualDenseBlock_5C(nn.Module):
        def __init__(self, nf=64, gc=32, bias=True):
            super(ResidualDenseBlock_5C, self).__init__()
            self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
            self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
            self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
            self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
            self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x


    class RRDB(nn.Module):
        def __init__(self, nf, gc=32):
            super(RRDB, self).__init__()
            self.RDB1 = ResidualDenseBlock_5C(nf, gc)
            self.RDB2 = ResidualDenseBlock_5C(nf, gc)
            self.RDB3 = ResidualDenseBlock_5C(nf, gc)

        def forward(self, x):
            out = self.RDB1(x)
            out = self.RDB2(out)
            out = self.RDB3(out)
            return out * 0.2 + x


    class RRDBNet(nn.Module):
        def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
            super(RRDBNet, self).__init__()
            self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
            self.RRDB_trunk = make_layer(RRDB, nb, nf=nf, gc=gc)
            self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            fea = self.conv_first(x)
            trunk = self.trunk_conv(self.RRDB_trunk(fea))
            fea = fea + trunk
            fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
            out = self.conv_last(self.lrelu(self.HRconv(fea)))
            return out


    AI_AVAILABLE = True
except ImportError:
    print("⚠️ Torch not found. Running in CPU-only Legacy Mode.")
    AI_AVAILABLE = False


# ===== UTILS =====

def emit_progress(progress, message="Processing..."):
    global current_job_id
    if current_job_id:
        socketio.emit('upscale_progress', {'job_id': current_job_id, 'progress': progress, 'message': message})
        print(f"📊 Progress: {progress}% - {message}")


def load_ai_model():
    global model, device
    if not AI_AVAILABLE: return None
    if model is not None: return model

    print("🔄 Loading Real-ESRGAN Model...")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"✅ Using Device: {device}")

    # Ensure directory exists
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)

    if not os.path.exists(WEIGHTS_PATH):
        print("⬇️ Downloading Real-ESRGAN x4plus weights...")
        subprocess.run(
            ['curl', '-L', 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
             '-o', WEIGHTS_PATH])

    try:
        net = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
        checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
        state_dict = checkpoint['params_ema'] if 'params_ema' in checkpoint else checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            if 'body.' in new_k: new_k = new_k.replace('body.', 'RRDB_trunk.')
            if 'conv_body' in new_k: new_k = new_k.replace('conv_body', 'trunk_conv')
            if 'conv_RRDB_trunk' in new_k: new_k = new_k.replace('conv_RRDB_trunk', 'trunk_conv')
            new_k = new_k.replace('rdb1', 'RDB1').replace('rdb2', 'RDB2').replace('rdb3', 'RDB3')
            if 'conv_up1' in new_k: new_k = new_k.replace('conv_up1', 'upconv1')
            if 'conv_up2' in new_k: new_k = new_k.replace('conv_up2', 'upconv2')
            if 'conv_hr' in new_k: new_k = new_k.replace('conv_hr', 'HRconv')
            new_state_dict[new_k] = v

        net.load_state_dict(new_state_dict, strict=True)
        net.eval()
        net = net.to(device)
        model = net
        print("✅ Real-ESRGAN Model Loaded Successfully!")
        return model
    except Exception as e:
        print(f"❌ Failed to load AI model: {e}")
        return None


# ===== TILING FUNCTION (FIXES MEMORY ERROR) =====
def tile_process(img, net, tile_size=400, tile_pad=10, scale=4):
    batch, channel, height, width = img.shape
    output_height = height * scale
    output_width = width * scale
    output_shape = (batch, channel, output_height, output_width)

    output = torch.zeros(output_shape, device=device)

    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    for y in range(tiles_y):
        for x in range(tiles_x):
            ofs_x = x * tile_size
            ofs_y = y * tile_size

            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_size, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_size, height)

            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            if input_end_x_pad - input_start_x_pad <= 0 or input_end_y_pad - input_start_y_pad <= 0:
                continue

            input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            with torch.no_grad():
                try:
                    output_tile = net(input_tile)
                except Exception as e:
                    print(f"Tile error: {e}")
                    return None

            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            output_start_x_pad = input_start_x_pad * scale
            output_end_x_pad = input_end_x_pad * scale
            output_start_y_pad = input_start_y_pad * scale
            output_end_y_pad = input_end_y_pad * scale

            tile_output_start_x = (input_start_x - input_start_x_pad) * scale
            tile_output_end_x = tile_output_start_x + (input_end_x - input_start_x) * scale
            tile_output_start_y = (input_start_y - input_start_y_pad) * scale
            tile_output_end_y = tile_output_start_y + (input_end_y - input_start_y) * scale

            output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
                output_tile[:, :, tile_output_start_y:tile_output_end_y, tile_output_start_x:tile_output_end_x]

    return output


def run_real_esrgan(img_cv):
    try:
        net = load_ai_model()
        if net is None: return None

        img = img_cv.astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)

        output = tile_process(img, net, tile_size=200, scale=4)

        if output is None: return None

        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        return (output * 255.0).round().astype(np.uint8)
    except Exception as e:
        print(f"⚠️ AI Inference Failed: {e}")
        return None


# ===== COLOR GRADING HELPER =====
def apply_color_science(img_pil, preset):
    if preset == 'balanced': return img_pil
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    if preset == 'vibrant':
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.35, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.10, 0, 255)
        hsv = hsv.astype(np.uint8)
        img_cv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        img_cv = cv2.detailEnhance(img_cv, sigma_s=8, sigma_r=0.10)
        result = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        return ImageEnhance.Color(
            ImageEnhance.Contrast(ImageEnhance.Sharpness(result).enhance(1.30)).enhance(1.15)).enhance(1.15)

    elif preset == 'cinematic':
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)
        hsv = hsv.astype(np.uint8)
        img_cv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        img_cv = cv2.detailEnhance(img_cv, sigma_s=10, sigma_r=0.12)
        result = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        return ImageEnhance.Contrast(ImageEnhance.Sharpness(result).enhance(1.15)).enhance(1.05)

    elif preset == 'crisp':
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img_cv = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        img_cv = cv2.detailEnhance(img_cv, sigma_s=10, sigma_r=0.15)
        result = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        return ImageEnhance.Sharpness(result).enhance(1.25)

    return img_pil


# ===== FALLBACK (Enhanced Lanczos) =====
def enhanced_lanczos(img, scale=4, preset='balanced'):
    w, h = img.size
    src_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    if preset == 'vibrant':
        den = src_bgr
    else:
        den = cv2.bilateralFilter(src_bgr, d=9, sigmaColor=60, sigmaSpace=60)

    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    limit = 3.0 if preset == 'vibrant' else (2.2 if preset == 'crisp' else 1.5)
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    den = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    base = Image.fromarray(cv2.cvtColor(den, cv2.COLOR_BGR2RGB))
    up = base.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    return apply_color_science(up, preset)


def smart_upscale_frame(frame_cv, target_w, target_h, preset='balanced'):
    current_h, current_w = frame_cv.shape[:2]
    scale = target_w / current_w

    if AI_AVAILABLE and scale >= 2.0:
        try:
            upscaled_4x = run_real_esrgan(frame_cv)
            if upscaled_4x is not None:
                final_frame = cv2.resize(upscaled_4x, (target_w, target_h), interpolation=cv2.INTER_AREA)
                final_pil = Image.fromarray(cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB))
                final_pil = apply_color_science(final_pil, preset)
                return cv2.cvtColor(np.array(final_pil), cv2.COLOR_RGB2BGR), "AI + " + preset.title()
        except Exception as e:
            print(f"⚠️ AI Frame failed: {e}")

    frame_pil = Image.fromarray(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB))
    int_scale = max(2, int(math.ceil(scale)))
    upscaled_pil = enhanced_lanczos(frame_pil, int_scale, preset)
    if upscaled_pil.size != (target_w, target_h):
        upscaled_pil = upscaled_pil.resize((target_w, target_h), Image.LANCZOS)

    return cv2.cvtColor(np.array(upscaled_pil), cv2.COLOR_RGB2BGR), "Lanczos"


# ===== EXTERNAL TOOLS PATH FIX =====
def get_ffmpeg_path():
    # Check for bundled ffmpeg in 'bin' folder
    bundled_path = resource_path(os.path.join('bin', 'ffmpeg'))
    if os.path.exists(bundled_path):
        return bundled_path
    return 'ffmpeg'  # Fallback to system ffmpeg


def get_rife_path():
    # Check for bundled rife-video in 'bin' folder
    bundled_path = resource_path(os.path.join('bin', 'rife-video'))
    if os.path.exists(bundled_path):
        return bundled_path
    return 'rife-video'  # Fallback to system rife-video


def merge_audio(original, processed, final):
    print("🎵 Merging audio...")
    ffmpeg_cmd = get_ffmpeg_path()
    cmd = [ffmpeg_cmd, '-y', '-i', processed, '-i', original, '-c:v', 'copy', '-c:a', 'copy', '-map', '0:v:0', '-map',
           '1:a:0?', '-shortest', final]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as e:
        print(f"⚠️ Audio merge failed: {e}")
        shutil.copy(processed, final)
        return False


def calculate_target_resolution(w, h, target):
    res = {'1080p': 1920, '2k': 2560, '4k': 3840}
    target_w = res.get(target, 1920)
    scale = target_w / w
    target_h = int(h * scale)
    return target_w, target_h, scale


# ===== ROUTES =====

@app.route('/upscale', methods=['POST'])
def upscale_image():
    global current_job_id
    try:
        data = request.get_json()
        current_job_id = data.get('job_id')
        img_bytes = base64.b64decode(data['image'])
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        scale = int(data.get('scale', 4))
        preset = data.get('preset', 'balanced')

        emit_progress(5, "Analyzing...")
        method = "Enhanced Lanczos"
        result = None

        if AI_AVAILABLE:
            try:
                emit_progress(20, "Running Real-ESRGAN AI...")
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                output_cv = run_real_esrgan(img_cv)
                if output_cv is not None:
                    result = Image.fromarray(cv2.cvtColor(output_cv, cv2.COLOR_BGR2RGB))
                    result = apply_color_science(result, preset)
                    method = f"Real-ESRGAN + {preset.title()}"
            except Exception as e:
                print(f"AI Failed: {e}")

        if result is None:
            emit_progress(20, "Running Fallback...")
            result = enhanced_lanczos(img, scale, preset)

        buffer = io.BytesIO()
        result.save(buffer, format='PNG')
        b64 = base64.b64encode(buffer.getvalue()).decode()

        emit_progress(100, "Done!")
        return jsonify({'success': True, 'upscaled_image': b64, 'method': method})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/upscale-video', methods=['POST'])
def upscale_video():
    global current_job_id
    try:
        video_file = request.files.get('video')
        target_format = request.form.get('target_format', '1080p')
        preset = request.form.get('preset', 'balanced')
        current_job_id = request.form.get('job_id')

        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, video_file.filename)
        video_file.save(input_path)

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        target_w, target_h, scale = calculate_target_resolution(w, h, target_format)

        temp_out = os.path.join(temp_dir, "temp.mp4")
        out = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_w, target_h))

        count = 0
        used_method = "Mixed"
        emit_progress(5, f"Upscaling to {target_w}x{target_h}...")

        while True:
            ret, frame = cap.read()
            if not ret: break

            final_frame, method = smart_upscale_frame(frame, target_w, target_h, preset)
            out.write(final_frame)
            used_method = method

            count += 1
            if count % 5 == 0:
                emit_progress(int((count / total) * 90), f"Frame: {count}/{total}")

        cap.release()
        out.release()

        final_out = os.path.join(temp_dir, "final.mp4")
        merge_audio(input_path, temp_out, final_out)

        with open(final_out, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()

        shutil.rmtree(temp_dir)
        emit_progress(100, "Complete!")
        return jsonify({'success': True, 'video': b64, 'method': used_method})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/interpolate', methods=['POST'])
def interpolate():
    global current_job_id
    try:
        video_file = request.files.get('video')
        target_fps = int(request.form.get('target_fps', 60))
        current_job_id = request.form.get('job_id')

        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, video_file.filename)
        video_file.save(input_path)

        cap = cv2.VideoCapture(input_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        ratio = target_fps / orig_fps if orig_fps > 0 else 2.0
        exp = 1 if ratio <= 2 else (2 if ratio <= 4 else 3)

        rife_out = os.path.join(temp_dir, "rife.mp4")

        rife_cmd = get_rife_path()
        cmd = [rife_cmd, f"--exp={exp}", f"--video={video_file.filename}", f"--fps={target_fps}",
               f"--output=rife.mp4", f"--model={MODEL_DIR}"]

        emit_progress(20, "Running RIFE...")
        subprocess.run(cmd, cwd=temp_dir, check=True)

        final_out = os.path.join(temp_dir, "final.mp4")
        merge_audio(input_path, rife_out, final_out)

        with open(final_out, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()

        shutil.rmtree(temp_dir)
        emit_progress(100, "Complete!")
        return jsonify({'success': True, 'video': b64})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    gpu_status = False
    if AI_AVAILABLE:
        gpu_status = torch.cuda.is_available() or torch.backends.mps.is_available()
    return jsonify({'status': 'ok', 'gpu_available': gpu_status, 'ai_available': AI_AVAILABLE})


if __name__ == '__main__':
    print("🚀 ULTIMATE AI SERVER STARTED")
    socketio.run(app, host='localhost', port=5001, debug=False)
