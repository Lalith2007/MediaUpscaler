import cv2
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

class ImageUpscaler:
    def __init__(self, model_path='weights/RealESRGAN_x4plus.pth'):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4, model_path=model_path,
            model=model, tile=400, tile_pad=10, pre_pad=0, half=True
        )
        return upsampler

    def upscale(self, input_path, scale=4, output_path=None):
        img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        output, _ = self.model.enhance(img, outscale=scale)
        if output_path:
            cv2.imwrite(str(output_path), output)
        return output
