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
import subprocess  # for calling RIFE
import sys         # to use current venv Python for RIFE if needed

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

GPU_AVAILABLE = False
REALESRGAN_AVAILABLE = False
upsampler = None
current_job_id = None

# Absolute path to RIFE weights directory (where flownet.pkl lives)
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'train_log')

# ===== Real-ESRGAN setup =====

try:
    import torch

    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("⚠️  No GPU detected - will use CPU mode")

    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

    REALESRGAN_AVAILABLE = True
    print("✅ Real-ESRGAN library loaded")
except Exception as e:
    print(f"⚠️  Real-ESRGAN not available: {e}")
    print("   Will use Enhanced LANCZOS fallback")


def emit_progress(progress, message="Processing..."):
    global current_job_id
    if current_job_id:
        socketio.emit('upscale_progress', {
            'job_id': current_job_id,
            'progress': progress,
            'message': message
        })
        print(f"📊 Progress: {progress}% - {message}")


def init_realesrgan():
    global upsampler
    if upsampler:
        return upsampler
    if not REALESRGAN_AVAILABLE or not GPU_AVAILABLE:
        return None
    try:
        emit_progress(10, "Loading AI model...")
        print("🔄 Loading Real-ESRGAN AI model...")
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4
        )
        upsampler = RealESRGANer(
            scale=4,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True,
            device='cuda'
        )
        emit_progress(20, "AI model loaded")
        print("✅ Real-ESRGAN loaded on GPU")
        return upsampler
    except Exception as e:
        print(f"❌ Real-ESRGAN init failed: {e}")
        return None


# ===== Enhanced LANCZOS upscaler =====

def enhanced_lanczos(img, scale=4, preset='balanced'):
    """Enhanced LANCZOS with presets - works for both images and video frames"""
    w, h = img.size

    if preset in ['cinematic', 'vibrant']:
        img_orig_np = np.array(img)

    emit_progress(18, "Pre-processing...")
    src_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    if preset == 'vibrant':
        den = src_bgr
    else:
        den = cv2.bilateralFilter(src_bgr, d=9, sigmaColor=60, sigmaSpace=60)

    if preset == 'crisp':
        emit_progress(26, "Boosting contrast...")
        lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        den = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
        emit_progress(30, "Enhancing details...")
        den = cv2.detailEnhance(den, sigma_s=10, sigma_r=0.15)
        g_sharp, g_contrast, g_color = 1.12, 1.06, 1.04
    elif preset == 'cinematic':
        emit_progress(26, "Boosting contrast (gentle)...")
        lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        den = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
        emit_progress(30, "Enhancing details (color-safe)...")
        den = cv2.detailEnhance(den, sigma_s=12, sigma_r=0.12)
        g_sharp, g_contrast, g_color = 1.10, 1.03, 1.02
    elif preset == 'vibrant':
        emit_progress(26, "Edge-preserving mode...")
        lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        den = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
        # sharpness/contrast/color tuned later
    else:
        # balanced
        g_sharp, g_contrast, g_color = 1.08, 1.04, 1.03

    emit_progress(36, "Upscaling...")
    base = Image.fromarray(cv2.cvtColor(den, cv2.COLOR_BGR2RGB))
    up = base.resize((w * scale, h * scale), Image.LANCZOS)

    if preset != 'vibrant':
        emit_progress(46, "Building edge mask...")
        gray = cv2.cvtColor(np.array(up), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        mask = (edges.astype(np.float32) / 255.0)[..., None]

        emit_progress(58, "Edge sharpening...")
        if preset == 'crisp':
            usm = up.filter(ImageFilter.UnsharpMask(radius=1.15, percent=170, threshold=3))
        elif preset == 'cinematic':
            usm = up.filter(ImageFilter.UnsharpMask(radius=1.2, percent=160, threshold=3))
        else:
            usm = up.filter(ImageFilter.UnsharpMask(radius=1.4, percent=130, threshold=4))

        up_np = np.array(up).astype(np.float32)
        usm_np = np.array(usm).astype(np.float32)
        blended = (usm_np * mask + up_np * (1.0 - mask)).clip(0, 255).astype(np.uint8)
        result = Image.fromarray(blended)
    else:
        result = up

    if preset == 'vibrant':
        emit_progress(50, "Frequency domain sharpening...")
        result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(result_np, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_channel = lab[:, :, 0]
        blurred = cv2.GaussianBlur(l_channel, (0, 0), 3)
        high_freq = l_channel - blurred
        sharpened = l_channel + high_freq * 2.5
        lab[:, :, 0] = np.clip(sharpened, 0, 255)
        result_np = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        result = Image.fromarray(cv2.cvtColor(result_np, cv2.COLOR_BGR2RGB))

        emit_progress(56, "VIBRANT COLOR MAXIMUM (HSV)...")
        img_orig_up = Image.fromarray(img_orig_np).resize((w * scale, h * scale), Image.LANCZOS)

        result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        result_hsv = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

        orig_bgr = cv2.cvtColor(np.array(img_orig_up), cv2.COLOR_RGB2BGR)
        orig_hsv = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

        result_hsv[:, :, 0] = orig_hsv[:, :, 0]
        result_hsv[:, :, 1] = np.clip(orig_hsv[:, :, 1] * 1.40, 0, 255)
        result_hsv[:, :, 2] = np.clip(orig_hsv[:, :, 2] * 1.10, 0, 255)

        result_hsv = result_hsv.astype(np.uint8)
        result_bgr = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)
        result = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))

        emit_progress(64, "ULTRA detail enhancement...")
        result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        result_np = cv2.detailEnhance(result_np, sigma_s=8, sigma_r=0.10)
        result = Image.fromarray(cv2.cvtColor(result_np, cv2.COLOR_BGR2RGB))

        emit_progress(70, "Global tuning...")
        result = ImageEnhance.Sharpness(result).enhance(1.45)
        result = ImageEnhance.Contrast(result).enhance(1.20)

        g_sharp, g_contrast, g_color = 1.20, 1.20, 1.15

    elif preset == 'cinematic':
        emit_progress(70, "Restoring color grading (HSV)...")
        img_orig_up = Image.fromarray(img_orig_np).resize((w * scale, h * scale), Image.LANCZOS)
        img_orig_up_hsv = cv2.cvtColor(
            cv2.cvtColor(np.array(img_orig_up), cv2.COLOR_RGB2BGR),
            cv2.COLOR_BGR2HSV
        ).astype(np.float32)
        result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        result_hsv = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        result_hsv[:, :, 0] = img_orig_up_hsv[:, :, 0]
        result_hsv[:, :, 1] = img_orig_up_hsv[:, :, 1] * 0.98
        result_hsv = np.clip(result_hsv, 0, 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)
        result = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))

        emit_progress(76, "Global tuning...")
        result = ImageEnhance.Sharpness(result).enhance(g_sharp)
        result = ImageEnhance.Contrast(result).enhance(g_contrast)
        result = ImageEnhance.Color(result).enhance(g_color)
    else:
        emit_progress(76, "Global tuning...")
        result = ImageEnhance.Sharpness(result).enhance(g_sharp)
        result = ImageEnhance.Contrast(result).enhance(g_contrast)
        result = ImageEnhance.Color(result).enhance(g_color)

    emit_progress(84, "Final polish...")
    if preset == 'vibrant':
        final = ImageEnhance.Sharpness(result).enhance(1.20)
        final = ImageEnhance.Color(final).enhance(1.15)
    else:
        fin_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        fin_bgr = cv2.fastNlMeansDenoisingColored(
            fin_bgr, None, h=2, hColor=2,
            templateWindowSize=7, searchWindowSize=21
        )
        final = Image.fromarray(cv2.cvtColor(fin_bgr, cv2.COLOR_BGR2RGB))
        final = ImageEnhance.Sharpness(final).enhance(1.03)

    return final


def upscale_frame(frame_cv, scale=4, preset='balanced', use_gpu=False):
    """Upscale a single video frame using GPU or CPU"""
    frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    if use_gpu and GPU_AVAILABLE and REALESRGAN_AVAILABLE:
        try:
            model = init_realesrgan()
            if model:
                output, _ = model.enhance(frame_cv, outscale=scale)
                return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"⚠️  GPU frame upscale failed: {e}, using CPU fallback")

    upscaled_pil = enhanced_lanczos(frame_pil, scale, preset)
    upscaled_cv = cv2.cvtColor(np.array(upscaled_pil), cv2.COLOR_RGB2BGR)
    return upscaled_cv


# ===== Image upscale endpoint =====

@app.route('/upscale', methods=['POST'])
def upscale():
    global current_job_id
    try:
        print("\n" + "=" * 50)
        print("🎨 UPSCALE REQUEST")
        print("=" * 50)
        data = request.get_json()
        img_b64 = data['image']
        scale = int(data.get('scale', 4))
        preset = data.get('preset', 'balanced')
        current_job_id = data.get('job_id', 'unknown')

        emit_progress(5, "Starting...")
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        w, h = img.size
        print(f"📥 Input: {w}x{h} | Scale: {scale}x | Preset: {preset}")

        emit_progress(10, "Preparing...")
        if GPU_AVAILABLE and REALESRGAN_AVAILABLE:
            try:
                model = init_realesrgan()
                if model:
                    print("🚀 Using Real-ESRGAN AI")
                    emit_progress(30, "AI enhancement...")
                    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    output, _ = model.enhance(img_cv, outscale=scale)
                    result = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
                    method = "Real-ESRGAN (GPU)"
                    emit_progress(95, "AI complete")
                else:
                    raise Exception("Model init failed")
            except Exception as e:
                print(f"⚠️  Real-ESRGAN failed: {e}")
                emit_progress(15, "Using enhanced mode...")
                result = enhanced_lanczos(img, scale, preset=preset)
                method = f'Enhanced LANCZOS ({preset})'
        else:
            print("⚡ Using Enhanced LANCZOS")
            emit_progress(15, "Enhanced mode...")
            result = enhanced_lanczos(img, scale, preset=preset)
            method = f'Enhanced LANCZOS ({preset})'

        emit_progress(98, "Finalizing...")
        buffer = io.BytesIO()
        result.save(buffer, format='PNG', quality=100)
        result_b64 = base64.b64encode(buffer.getvalue()).decode()
        emit_progress(100, "Complete!")
        print(f"📤 Output: {result.size[0]}x{result.size[1]} | Method: {method}")
        print("=" * 50 + "\n")

        return jsonify({
            'success': True,
            'upscaled_image': result_b64,
            'size': f"{result.size[0]}x{result.size[1]}",
            'method': method,
            'gpu_used': GPU_AVAILABLE
        })
    except Exception as e:
        emit_progress(0, f"Error: {str(e)}")
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400
    finally:
        current_job_id = None


# ===== Video upscale endpoint =====

@app.route('/upscale-video', methods=['POST'])
def upscale_video():
    """Upscale video frames one by one with quality enhancement"""
    global current_job_id
    try:
        print("\n" + "=" * 50)
        print("🎬 VIDEO UPSCALE REQUEST")
        print("=" * 50)

        video_file = request.files.get('video')
        scale = int(request.form.get('scale', 2))
        preset = request.form.get('preset', 'balanced')
        current_job_id = request.form.get('job_id', 'unknown')

        if not video_file:
            return jsonify({'success': False, 'error': 'No video provided'}), 400

        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, video_file.filename)
        output_filename = f"upscaled_{scale}x_{int(time.time())}.mp4"
        output_path = os.path.join(temp_dir, output_filename)

        video_file.save(input_path)
        print(f"📥 Video uploaded: {video_file.filename}")

        emit_progress(5, "Opening video...")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            shutil.rmtree(temp_dir)
            return jsonify({'success': False, 'error': 'Cannot open video'}), 400

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        new_w, new_h = w * scale, h * scale

        print(f"📹 Video: {w}x{h} @ {fps}fps, {total_frames} frames → {new_w}x{new_h}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_w, new_h))

        frame_count = 0
        emit_progress(10, f'Processing {total_frames} frames...')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                upscaled = upscale_frame(
                    frame, scale, preset,
                    use_gpu=GPU_AVAILABLE and REALESRGAN_AVAILABLE
                )
                out.write(upscaled)
            except Exception as frame_err:
                print(f"⚠️  Frame {frame_count} error: {frame_err}, using basic upscale")
                upscaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                out.write(upscaled)

            frame_count += 1
            progress = int((frame_count / max(total_frames, 1)) * 85)
            if frame_count % 10 == 0 or frame_count == total_frames:
                emit_progress(progress, f'Frame {frame_count}/{total_frames}')
                print(f"📊 Progress: {progress}% ({frame_count}/{total_frames})")

        cap.release()
        out.release()

        if not os.path.exists(output_path):
            shutil.rmtree(temp_dir)
            return jsonify({'success': False, 'error': 'Failed to save video'}), 400

        emit_progress(90, "Encoding video...")
        file_size = os.path.getsize(output_path)
        print(f"✅ Video saved: {output_path} ({file_size:,} bytes)")

        with open(output_path, 'rb') as f:
            video_b64 = base64.b64encode(f.read()).decode()

        emit_progress(100, "Complete!")
        method = "Real-ESRGAN (GPU)" if GPU_AVAILABLE and REALESRGAN_AVAILABLE else f"Enhanced LANCZOS ({preset})"
        print(f"📤 Output: {output_filename} | Method: {method}")
        print("=" * 50 + "\n")

        shutil.rmtree(temp_dir)

        return jsonify({
            'success': True,
            'video': video_b64,
            'filename': output_filename,
            'method': method,
            'gpu_used': GPU_AVAILABLE and REALESRGAN_AVAILABLE,
            'frames_processed': frame_count
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        emit_progress(0, f"Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400
    finally:
        current_job_id = None


# ===== Legacy FrameInterpolator (kept, not used by RIFE) =====

class FrameInterpolator:
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

    def validate_video(self, video_path, max_duration=300):
        """Validate video file (max 5 minutes)"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, "Cannot open video file"

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0

            if duration > max_duration:
                return False, f"Video too long ({duration:.1f}s > {max_duration}s)"

            cap.release()
            return True, f"✅ Valid video: {duration:.1f}s @ {fps:.0f}fps"
        except Exception as e:
            return False, str(e)

    def interpolate_frames(self, frame1, frame2, num_interpolations=1):
        """Optical-flow interpolation (unused now, kept for reference)"""
        try:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )

            interpolated_frames = []
            h, w = frame1.shape[:2]
            x, y = np.meshgrid(np.arange(w), np.arange(h))

            for i in range(1, num_interpolations + 1):
                alpha = i / (num_interpolations + 1)
                scaled_flow = flow * alpha

                map_x = (x + scaled_flow[..., 0]).astype(np.float32)
                map_y = (y + scaled_flow[..., 1]).astype(np.float32)
                warped1 = cv2.remap(frame1, map_x, map_y, cv2.INTER_LINEAR)

                reverse_flow = flow * (1 - alpha)
                map_x_rev = (x - reverse_flow[..., 0]).astype(np.float32)
                map_y_rev = (y - reverse_flow[..., 1]).astype(np.float32)
                warped2 = cv2.remap(frame2, map_x_rev, map_y_rev, cv2.INTER_LINEAR)

                interpolated = cv2.addWeighted(warped1, 1 - alpha, warped2, alpha, 0)
                interpolated_frames.append(interpolated)

            return interpolated_frames
        except Exception as e:
            print(f"❌ Interpolation error: {e}")
            return []


interpolator = FrameInterpolator()


# ===== New RIFE-based interpolate endpoint =====

@app.route('/interpolate', methods=['POST'])
def interpolate_video():
    """
    Frame interpolation endpoint using RIFE via the rife-video CLI.
    No FPS restriction: will run even if OpenCV reports a strange FPS.
    """
    global current_job_id
    try:
        print("\n" + "=" * 50)
        print("🎬 RIFE FRAME INTERPOLATION REQUEST")
        print("=" * 50)

        video_file = request.files.get('video')
        target_fps = int(request.form.get('target_fps', 60))
        current_job_id = request.form.get('job_id', 'unknown')

        if not video_file:
            return jsonify({'success': False, 'error': 'No video file provided'}), 400

        # Check model weights exist (better error than FileNotFoundError inside rife)
        flownet_path = os.path.join(MODEL_DIR, 'flownet.pkl')
        if not os.path.exists(flownet_path):
            return jsonify({
                'success': False,
                'error': f'RIFE weights not found at {flownet_path}. Place flownet.pkl in train_log/.'
            }), 500

        # Temp paths
        temp_dir = tempfile.mkdtemp()

        # Save upload into temp_dir
        input_filename = video_file.filename
        input_path = os.path.join(temp_dir, input_filename)
        video_file.save(input_path)
        print(f"📥 Video uploaded: {video_file.filename}")
        print(f"📁 Stored at: {input_path}")

        # Explicit output filename in temp_dir
        output_filename = f"interpolated_rife_{target_fps}fps_{int(time.time())}.mp4"
        output_path = os.path.join(temp_dir, output_filename)

        emit_progress(5, "Analyzing video...")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            shutil.rmtree(temp_dir, ignore_errors=True)
            return jsonify({'success': False, 'error': 'Cannot open video'}), 400

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        if original_fps <= 0:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return jsonify({'success': False, 'error': 'Could not read FPS from video'}), 400

        print(f"📹 Video: {frame_width}x{frame_height} @ {original_fps:.3f}fps, {total_frames} frames")
        print(f"📤 Target: {target_fps}fps")

        # Compute ratio; clamp minimum to 1 so we always interpolate at least a bit
        raw_ratio = target_fps / original_fps
        ratio = max(raw_ratio, 1.0)

        if ratio <= 2:
            exp = 1       # 2x
        elif ratio <= 4:
            exp = 2       # 4x
        else:
            exp = 3       # 8x

        print(f"⚡ RIFE settings: exp={exp} (≈ {2 ** exp}x), raw_ratio={raw_ratio:.2f}, used_ratio={ratio:.2f}")
        emit_progress(10, f"Running RIFE (exp={exp}, target {target_fps}fps)...")

        # Call the rife-video CLI (installed in venv), using correct --model argument. [web:195]
        cmd = [
            "rife-video",
            f"--exp={exp}",
            f"--video={input_filename}",      # relative to cwd=temp_dir
            f"--fps={target_fps}",
            f"--output={output_filename}",    # explicit output file in temp_dir
            f"--model={MODEL_DIR}",           # directory containing flownet.pkl and other weights
        ]

        print("🔧 Command (cwd=temp_dir):", " ".join(cmd))
        emit_progress(20, "Starting RIFE interpolation...")

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=temp_dir
        )

        print("──── rife-video stdout ────")
        print(proc.stdout)
        print("──── rife-video stderr ────")
        print(proc.stderr)
        print("────────────────────────────")

        if proc.returncode != 0:
            emit_progress(0, "RIFE failed")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return jsonify({
                'success': False,
                'error': f"rife-video failed (code {proc.returncode}). See server logs for details."
            }), 500

        emit_progress(90, "Checking output video...")

        if not os.path.exists(output_path):
            emit_progress(0, "Error: Output file not created")
            print(f"❌ Expected output not found at {output_path}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return jsonify({'success': False, 'error': 'Output video not created'}), 500

        emit_progress(95, "Encoding final video...")

        with open(output_path, 'rb') as f:
            video_b64 = base64.b64encode(f.read()).decode('utf-8')

        emit_progress(100, "Complete!")
        print(f"✅ Interpolated video saved: {output_path}")
        print("=" * 50 + "\n")

        shutil.rmtree(temp_dir, ignore_errors=True)

        return jsonify({
            'success': True,
            'video': video_b64,
            'filename': output_filename,
            'target_fps': target_fps,
            'method': f'RIFE (exp={exp})'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        emit_progress(0, f"Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        current_job_id = None


# ===== Health endpoint =====

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'gpu_available': GPU_AVAILABLE,
        'realesrgan_available': REALESRGAN_AVAILABLE,
        'mode': 'GPU-Accelerated AI' if GPU_AVAILABLE and REALESRGAN_AVAILABLE else 'Enhanced LANCZOS',
        'presets': ['balanced', 'crisp', 'cinematic', 'vibrant']
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🎨 SMART AI UPSCALING & INTERPOLATION API SERVER")
    print("=" * 60)
    if GPU_AVAILABLE and REALESRGAN_AVAILABLE:
        print("🚀 Mode: GPU-Accelerated Real-ESRGAN")
        print("   Speed: Ultra-fast (2-8 seconds per frame)")
        print("   Quality: Maximum AI")
    else:
        print("⚡ Mode: Enhanced LANCZOS / Optical Flow / RIFE")
        print("   Presets: balanced | crisp | cinematic | vibrant")
        print("   Speed: Fast (5-15 seconds per frame)")
        print("   Quality: PICSART-LEVEL / Smooth Motion")
    print("   Features: Color boost, contrast, sharpness, denoising, frame interpolation (RIFE)")
    print("=" * 60 + "\n")
    socketio.run(app, host='localhost', port=5001, debug=False)
