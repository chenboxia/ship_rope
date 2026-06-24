"""
视频抽帧脚本
用法:
  python scripts/extract_frames.py --video 视频.mp4
  python scripts/extract_frames.py --video 视频.mp4 --interval 2 --output data/images/raw
  python scripts/extract_frames.py --video 视频.mp4 --start 60 --end 120
  python scripts/extract_frames.py --video 视频.mp4 --resize 1280x720

参数:
  --video     视频文件路径
  --interval  每隔几秒抽一帧（默认1秒）
  --output    输出目录（默认 data/images/raw）
  --start     起始秒数（默认0）
  --end       结束秒数（默认到视频结尾）
  --resize    缩放尺寸，格式 宽x高（默认不缩放）
  --quality   JPEG质量 1-100（默认95）
"""
import sys, os, argparse
import cv2


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="data/images/raw")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float, default=-1)
    parser.add_argument("--resize", type=str, default="")
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: cannot open video:", args.video)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Video: %s" % args.video)
    print("Resolution: %dx%d, FPS: %.1f, Duration: %.1fs" % (w, h, fps, duration))
    print("Interval: %.1fs" % args.interval)
    print("Output: %s" % args.output)

    resize = None
    if args.resize:
        rw, rh = args.resize.split("x")
        resize = (int(rw), int(rh))
        print("Resize: %dx%d" % resize)

    os.makedirs(args.output, exist_ok=True)

    start_frame = int(args.start * fps)
    end_frame = int(args.end * fps) if args.end > 0 else total_frames
    frame_interval = int(args.interval * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    count = 0
    saved = 0
    current_frame = start_frame

    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            if resize:
                frame = cv2.resize(frame, resize)
            filename = "frame_%06d.jpg" % saved
            path = os.path.join(args.output, filename)
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            saved += 1
            if saved % 50 == 0:
                print("  saved %d frames..." % saved)
        count += 1
        current_frame = start_frame + count

    cap.release()
    print("\nDone! Saved %d frames to %s" % (saved, args.output))


if __name__ == "__main__":
    main()
