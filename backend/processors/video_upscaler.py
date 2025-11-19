import cv2
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

class VideoUpscaler:
    def __init__(self, model_path='weights/RealESRGAN_x4plus.pth'):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4, model_path=model_path,
            model=model, tile=400, tile_pad=10, pre_pad=0, half=True
        )
        return upsampler

    def upscale(self, input_path, output_path, scale=4, progress_callback=None):
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()

        upscaled, _ = self.model.enhance(frame, outscale=4)
        h, w = upscaled.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        out.write(upscaled)

        frame_count = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            upscaled, _ = self.model.enhance(frame, outscale=4)
            out.write(upscaled)
            frame_count += 1
            if progress_callback:
                progress_callback(int((frame_count / total_frames) * 100))
        cap.release()
        out.release()
        return output_path
