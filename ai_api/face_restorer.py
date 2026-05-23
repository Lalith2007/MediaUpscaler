"""
Face Restoration Module — ONNX-based GFPGAN v1.4
Runs 100% locally using ONNX Runtime. No basicsr/facexlib required.
Uses OpenCV YuNet for face detection + GFPGAN ONNX for face restoration.
"""

import os
import cv2
import numpy as np
import subprocess

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ onnxruntime not installed. Face restoration disabled.")


# ===== PATHS =====
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(MODULE_DIR, 'weights')
YUNET_PATH = os.path.join(WEIGHTS_DIR, 'face_detection_yunet_2023mar.onnx')
GFPGAN_PATH = os.path.join(WEIGHTS_DIR, 'GFPGANv1.4.onnx')

# Model URLs
YUNET_URL = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
GFPGAN_URL = 'https://huggingface.co/Neus/GFPGANv1.4/resolve/main/GFPGANv1.4.onnx'

# ===== GLOBALS =====
_face_detector = None
_gfpgan_session = None


def download_model(url, dest_path):
    """Download a model file if it doesn't already exist."""
    if os.path.exists(dest_path):
        return True
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"⬇️  Downloading {os.path.basename(dest_path)}...")
    try:
        subprocess.run(
            ['curl', '-fSL', '--progress-bar', '-o', dest_path, url],
            check=True, timeout=300
        )
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
            print(f"✅ Downloaded {os.path.basename(dest_path)} ({os.path.getsize(dest_path)} bytes)")
            return True
        else:
            print(f"❌ Downloaded file too small or missing: {dest_path}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            return False
    except Exception as e:
        print(f"❌ Download failed: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def ensure_models():
    """Download both YuNet and GFPGAN models if needed."""
    ok1 = download_model(YUNET_URL, YUNET_PATH)
    ok2 = download_model(GFPGAN_URL, GFPGAN_PATH)
    return ok1 and ok2


def get_face_detector(img_width, img_height):
    """Get (or create) the OpenCV YuNet face detector."""
    global _face_detector
    if not os.path.exists(YUNET_PATH):
        return None
    _face_detector = cv2.FaceDetectorYN.create(
        YUNET_PATH, "", (img_width, img_height),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=10
    )
    return _face_detector


def get_gfpgan_session():
    """Get (or create) the ONNX Runtime session for GFPGAN."""
    global _gfpgan_session
    if _gfpgan_session is not None:
        return _gfpgan_session
    if not ONNX_AVAILABLE or not os.path.exists(GFPGAN_PATH):
        return None
    try:
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _gfpgan_session = ort.InferenceSession(GFPGAN_PATH, sess_opts,
                                                providers=['CPUExecutionProvider'])
        print("✅ GFPGAN ONNX session loaded successfully!")
        return _gfpgan_session
    except Exception as e:
        print(f"❌ Failed to load GFPGAN ONNX: {e}")
        return None


def align_face_simple(img, bbox, target_size=512):
    """
    Simple face alignment: crop the face region with padding,
    resize to target_size x target_size.
    Returns: (cropped_face, (x, y, w, h) in original image coords)
    """
    h, w = img.shape[:2]
    x1, y1, fw, fh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    # Add generous padding around the face (GFPGAN expects the full head)
    pad_w = int(fw * 0.7)
    pad_h = int(fh * 0.7)

    # Expand the crop region
    cx1 = max(0, x1 - pad_w)
    cy1 = max(0, y1 - pad_h)
    cx2 = min(w, x1 + fw + pad_w)
    cy2 = min(h, y1 + fh + pad_h)

    # Make it square (GFPGAN expects square input)
    crop_w = cx2 - cx1
    crop_h = cy2 - cy1
    side = max(crop_w, crop_h)

    # Center the square
    center_x = (cx1 + cx2) // 2
    center_y = (cy1 + cy2) // 2
    sx1 = max(0, center_x - side // 2)
    sy1 = max(0, center_y - side // 2)
    sx2 = min(w, sx1 + side)
    sy2 = min(h, sy1 + side)

    face_crop = img[sy1:sy2, sx1:sx2].copy()
    if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
        return None, None

    face_resized = cv2.resize(face_crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    return face_resized, (sx1, sy1, sx2, sy2)


def restore_face_onnx(face_img_512):
    """
    Run GFPGAN ONNX inference on a 512x512 face image.
    Returns: restored 512x512 face image (numpy BGR).
    """
    session = get_gfpgan_session()
    if session is None:
        return face_img_512

    # Preprocess: BGR -> RGB, normalize to [-1, 1], NCHW
    face_rgb = cv2.cvtColor(face_img_512, cv2.COLOR_BGR2RGB)
    face_input = face_rgb.astype(np.float32) / 255.0
    face_input = (face_input - 0.5) / 0.5  # Normalize to [-1, 1]
    face_input = np.transpose(face_input, (2, 0, 1))  # HWC -> CHW
    face_input = np.expand_dims(face_input, axis=0)  # Add batch dim

    try:
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: face_input})[0]

        # Postprocess: NCHW -> HWC, denormalize, clip, RGB -> BGR
        result = np.squeeze(result, axis=0)  # Remove batch
        result = np.transpose(result, (1, 2, 0))  # CHW -> HWC
        result = (result * 0.5 + 0.5) * 255.0  # Denormalize
        result = np.clip(result, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        return result
    except Exception as e:
        print(f"⚠️ GFPGAN inference error: {e}")
        return face_img_512


def blend_restored_face(original, restored_face, coords, blend_ratio=0.85):
    """
    Paste the restored face back onto the original image with smooth blending.
    """
    sx1, sy1, sx2, sy2 = coords
    region_h = sy2 - sy1
    region_w = sx2 - sx1

    if region_h <= 0 or region_w <= 0:
        return original

    # Resize restored face to match the original crop region
    restored_resized = cv2.resize(restored_face, (region_w, region_h), interpolation=cv2.INTER_LANCZOS4)

    # Create a smooth elliptical mask for natural blending
    mask = np.zeros((region_h, region_w), dtype=np.float32)
    center = (region_w // 2, region_h // 2)
    axes = (int(region_w * 0.42), int(region_h * 0.42))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

    # Apply Gaussian blur to soften the mask edges
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=region_w * 0.08)
    mask = np.clip(mask, 0, 1)

    # Apply blend ratio
    mask = mask * blend_ratio

    # Blend: result = mask * restored + (1 - mask) * original
    mask_3ch = np.stack([mask] * 3, axis=-1)
    original_region = original[sy1:sy2, sx1:sx2].astype(np.float32)
    restored_f = restored_resized.astype(np.float32)
    blended = mask_3ch * restored_f + (1 - mask_3ch) * original_region
    original[sy1:sy2, sx1:sx2] = blended.astype(np.uint8)

    return original


def restore_faces(img_cv, max_faces=5):
    """
    Main entry point: detect faces in an image and restore each one.
    Args:
        img_cv: Input image in BGR format (numpy array).
        max_faces: Maximum number of faces to process.
    Returns:
        result: Image with restored faces (BGR numpy array).
        face_count: Number of faces detected and restored.
    """
    if not ONNX_AVAILABLE:
        print("⚠️ ONNX Runtime not available. Skipping face restoration.")
        return img_cv, 0

    if not ensure_models():
        print("⚠️ Face restoration models not available. Skipping.")
        return img_cv, 0

    h, w = img_cv.shape[:2]
    detector = get_face_detector(w, h)
    if detector is None:
        return img_cv, 0

    # Detect faces
    _, faces = detector.detect(img_cv)
    if faces is None or len(faces) == 0:
        print("ℹ️ No faces detected in image.")
        return img_cv, 0

    face_count = min(len(faces), max_faces)
    print(f"👤 Detected {len(faces)} face(s), processing {face_count}...")

    result = img_cv.copy()

    for i in range(face_count):
        bbox = faces[i]

        # Align and crop face
        face_crop, coords = align_face_simple(result, bbox, target_size=512)
        if face_crop is None:
            continue

        # Restore the face
        restored = restore_face_onnx(face_crop)

        # Blend back into the image
        result = blend_restored_face(result, restored, coords)
        print(f"  ✅ Face {i+1}/{face_count} restored")

    return result, face_count


def is_available():
    """Check if face restoration is available."""
    return ONNX_AVAILABLE and os.path.exists(GFPGAN_PATH) and os.path.exists(YUNET_PATH)
