"""
实时推理引擎
支持摄像头/视频文件输入，工业级推理流程。
"""
import os
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger
from src.monitor import MonitorPipeline


class InferenceEngine:
    """实时推理引擎"""

    def __init__(self, config: dict):
        self.config = config
        self.pipeline = MonitorPipeline(config)
        self.output_cfg = config.get("output", {})
        self.save_video = self.output_cfg.get("save_video", True)
        self.save_dir = self.output_cfg.get("save_dir", "outputs/")
        self.snapshot_on_alarm = self.output_cfg.get("snapshot_on_alarm", True)
        self.snapshot_dir = self.output_cfg.get("snapshot_dir", "outputs/snapshots/")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.writer = None
        self.frame_count = 0
        self.alarm_count = 0
        self.fps_history = []
        self._last_frame_time = None

    def run(self, source, max_frames: int = 0, show: bool = False):
        """
        运行推理。
        source: 视频文件路径、摄像头索引(int)或RTSP地址
        max_frames: 最大处理帧数，0表示不限制
        show: 是否显示实时画面（需要GUI环境）
        """
        cap = self._open_source(source)
        if cap is None or not cap.isOpened():
            logger.error(f"Failed to open source: {source}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Source opened: {w}x{h} @ {fps:.1f}fps")
        if self.save_video:
            self._init_writer(w, h, fps)
        self._last_frame_time = time.time()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of stream")
                    break
                result = self.pipeline.process_frame(frame)
                self.frame_count += 1
                # FPS计算
                now = time.time()
                dt = now - self._last_frame_time
                self._last_frame_time = now
                if dt > 0:
                    self.fps_history.append(1.0 / dt)
                    if len(self.fps_history) > 100:
                        self.fps_history.pop(0)
                # 报警处理
                if result["alarms"]:
                    for alarm in result["alarms"]:
                        self.alarm_count += 1
                        logger.warning(f"ALARM #{self.alarm_count}: {alarm['message']}")
                        if self.snapshot_on_alarm:
                            self._save_snapshot(result["annotated_frame"], alarm)
                # 写入输出视频
                if self.writer is not None:
                    self.writer.write(result["annotated_frame"])
                # 显示
                if show:
                    cv2.imshow("Mooring Monitor", result["annotated_frame"])
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("User quit")
                        break
                # 帧数限制
                if max_frames > 0 and self.frame_count >= max_frames:
                    logger.info(f"Reached max frames: {max_frames}")
                    break
                if self.frame_count % 100 == 0:
                    avg_fps = np.mean(self.fps_history) if self.fps_history else 0
                    logger.info(f"Processed {self.frame_count} frames, "
                                f"avg FPS: {avg_fps:.1f}, alarms: {self.alarm_count}")
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            cap.release()
            if self.writer is not None:
                self.writer.release()
            if show:
                cv2.destroyAllWindows()
            self._print_summary()

    def _open_source(self, source):
        """打开视频源"""
        if isinstance(source, int):
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        elif isinstance(source, str):
            if source.startswith("rtsp://") or source.startswith("http://"):
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            return cv2.VideoCapture(source)
        return None

    def _init_writer(self, w: int, h: int, fps: float):
        """初始化视频写入器"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(self.save_dir, f"monitor_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        logger.info(f"Output video: {out_path}")

    def _save_snapshot(self, frame: np.ndarray, alarm: dict):
        """保存报警快照"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(self.snapshot_dir,
                            f"alarm_{alarm['rope_id']}_{timestamp}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info(f"Snapshot saved: {path}")

    def _print_summary(self):
        """打印运行摘要"""
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        logger.info("=" * 50)
        logger.info("Session Summary:")
        logger.info(f"  Total frames: {self.frame_count}")
        logger.info(f"  Average FPS: {avg_fps:.1f}")
        logger.info(f"  Total alarms: {self.alarm_count}")
        logger.info("=" * 50)
