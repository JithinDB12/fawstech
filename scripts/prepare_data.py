import os
import argparse
from glob import glob
from deepfake_detector.utils.video import extract_frames_ffmpeg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_dir', type=str, required=True, help='Directory with input videos')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save extracted frames')
    parser.add_argument('--fps', type=float, default=5.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    video_paths = []
    for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv'):
        video_paths.extend(glob(os.path.join(args.videos_dir, '**', ext), recursive=True))
    for vp in video_paths:
        rel = os.path.relpath(vp, args.videos_dir)
        video_id = os.path.splitext(rel)[0].replace(os.sep, '_')
        out_dir = os.path.join(args.out_dir, video_id)
        print(f'Extracting {vp} -> {out_dir}')
        extract_frames_ffmpeg(vp, out_dir, fps=args.fps, overwrite=False)


if __name__ == '__main__':
    main()
