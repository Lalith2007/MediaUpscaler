import cv2
import numpy as np
import os
from pathlib import Path


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
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            if duration > max_duration:
                return False, f"Video too long ({duration:.1f}s > {max_duration}s)"

            cap.release()
            return True, f"✅ Valid video: {duration:.1f}s @ {fps:.0f}fps"
        except Exception as e:
            return False, str(e)

    def interpolate_frames(self, frame1, frame2, num_interpolations=1):
        """
        Interpolate frames between two consecutive frames using optical flow.
        num_interpolations: number of frames to generate between frame1 and frame2
        """
        try:
            # Convert to grayscale for optical flow
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # Correct Farneback call: 10 arguments only
            flow = cv2.calcOpticalFlowFarneback(
                gray1,      # prev
                gray2,      # next
                None,       # flow
                0.5,        # pyr_scale
                3,          # levels
                15,         # winsize
                3,          # iterations
                5,          # poly_n
                1.2,        # poly_sigma
                0           # flags
            )

            interpolated_frames = []

            h, w = frame1.shape[:2]
            x, y = np.meshgrid(np.arange(w), np.arange(h))

            for i in range(1, num_interpolations + 1):
                # Weight for interpolation (0 to 1)
                alpha = i / (num_interpolations + 1)

                # Scale optical flow by alpha
                scaled_flow = flow * alpha

                # Apply flow to frame1 using remap
                map_x = (x + scaled_flow[..., 0]).astype(np.float32)
                map_y = (y + scaled_flow[..., 1]).astype(np.float32)
                warped1 = cv2.remap(frame1, map_x, map_y, cv2.INTER_LINEAR)

                # Reverse warp frame2
                reverse_flow = flow * (1 - alpha)
                map_x_rev = (x - reverse_flow[..., 0]).astype(np.float32)
                map_y_rev = (y - reverse_flow[..., 1]).astype(np.float32)
                warped2 = cv2.remap(frame2, map_x_rev, map_y_rev, cv2.INTER_LINEAR)

                # Blend warped frames
                interpolated = cv2.addWeighted(warped1, 1 - alpha, warped2, alpha, 0)
                interpolated_frames.append(interpolated)

            return interpolated_frames
        except Exception as e:
            print(f"❌ Interpolation error: {e}")
            return []

    def process_video(self, input_path, output_path, target_fps, progress_callback=None):
        """
        Process video to increase FPS using frame interpolation.
        """
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return False, "Cannot open video"

            # Get video properties
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(f"📥 Input: {original_fps:.0f}fps, {total_frames} frames, {frame_width}x{frame_height}")
            print(f"📤 Target: {target_fps}fps")

            # Calculate interpolation factor
            if target_fps <= original_fps:
                return False, f"Target FPS ({target_fps}) must be > original FPS ({original_fps:.0f})"

            interp_factor = target_fps / original_fps
            frames_to_insert = int(interp_factor) - 1

            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, target_fps, (frame_width, frame_height))

            frame_count = 0
            prev_frame = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Write original frame
                out.write(frame)

                # Generate and write interpolated frames
                if prev_frame is not None and frames_to_insert > 0:
                    interpolated = self.interpolate_frames(prev_frame, frame, frames_to_insert)
                    for inter_frame in interpolated:
                        out.write(inter_frame)

                prev_frame = frame
                frame_count += 1

                # Progress callback
                if progress_callback:
                    progress = int((frame_count / total_frames) * 100)
                    progress_callback(progress, f"Processing frame {frame_count}/{total_frames}")

            cap.release()
            out.release()

            return True, f"✅ Video saved: {target_fps}fps @ {frame_width}x{frame_height}"
        except Exception as e:
            return False, str(e)


if __name__ == "__main__":
    interpolator = FrameInterpolator()
    print("✅ Frame Interpolator initialized")
