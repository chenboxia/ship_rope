"""
壁面区域标定工具
用法:
  python scripts/calibrate.py --source video.mp4       # 从视频标定
  python scripts/calibrate.py --source 0               # 从摄像头标定
  python scripts/calibrate.py --image frame.jpg        # 从图片标定

操作说明:
  左键点击: 添加标定点（至少3个点构成多边形）
  右键点击: 撤销上一个点
  按 s 键: 保存标定结果
  按 r 键: 重置所有点
  按 q 键: 退出
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import cv2
import yaml
import numpy as np
from loguru import logger


points = []


def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        logger.info(f"Point added: ({x}, {y}), total: {len(points)}")
    elif event == cv2.EVENT_RBUTTONDOWN and points:
        removed = points.pop()
        logger.info(f"Point removed: {removed}, total: {len(points)}")


def draw_overlay(frame, pts):
    vis = frame.copy()
    for i, pt in enumerate(pts):
        cv2.circle(vis, tuple(pt), 6, (0, 0, 255), -1)
        cv2.putText(vis, str(i + 1), (pt[0] + 8, pt[1] - 8),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if len(pts) >= 3:
        arr = np.array(pts, dtype=np.int32)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [arr], (255, 200, 0))
        vis = cv2.addWeighted(overlay, 0.2, vis, 0.8, 0)
        cv2.polylines(vis, [arr], True, (255, 200, 0), 2)
    elif len(pts) == 2:
        cv2.line(vis, tuple(pts[0]), tuple(pts[1]), (255, 200, 0), 2)
    # 提示文字
    cv2.putText(vis, "L-click: add  R-click: undo  S: save  R: reset  Q: quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(vis, f"Points: {len(pts)}  (need >= 3 to save)",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    return vis


def save_calibration(pts, output_path):
    """保存标定结果到config.yaml"""
    with open(output_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["wall_region"]["points"] = pts
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    logger.info(f"Calibration saved to {output_path}")
    logger.info(f"Wall region points: {pts}")


def main():
    global points
    parser = argparse.ArgumentParser(description="Calibrate wall region")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=str, help="Video file or camera index")
    group.add_argument("--image", type=str, help="Single image file")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Config file to save calibration")
    args = parser.parse_args()

    frame = None
    cap = None

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            logger.error(f"Failed to load image: {args.image}")
            return
    else:
        source = args.source
        try:
            source = int(source)
        except ValueError:
            pass
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Failed to open source: {args.source}")
            return
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame")
            return

    # 加载已有标定点
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        existing = config.get("wall_region", {}).get("points", [])
        if existing:
            points = [list(p) for p in existing]
            logger.info(f"Loaded {len(points)} existing calibration points")
    except Exception:
        pass

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibration", mouse_callback)
    logger.info("Calibration tool started")
    logger.info("Select wall region points on the frame")

    while True:
        vis = draw_overlay(frame, points)
        cv2.imshow("Calibration", vis)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            if len(points) >= 3:
                save_calibration(points, args.config)
                logger.info("Calibration saved successfully")
            else:
                logger.warning("Need at least 3 points to save")
        elif key == ord("r"):
            points.clear()
            logger.info("Points reset")
        elif key == ord("n") and cap is not None:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video, rewinding")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
