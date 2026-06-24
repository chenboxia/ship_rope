"""
视频流管理模块
支持多路摄像头并行采集、断流自动重连、心跳检测。
"""
import time
import threading
import cv2
import numpy as np
from typing import Optional, Callable
from loguru import logger


class CameraStream:
    """单路摄像头流，带自动重连和心跳"""

    def __init__(self, source, name: str = "cam0",
                 reconnect_interval: float = 5.0,
                 max_reconnect: int = 0,
                 heartbeat_timeout: float = 10.0,
                 buffer_size: int = 1):
        self.source = source
        self.name = name
        self.reconnect_interval = reconnect_interval
        self.max_reconnect = max_reconnect
        self.heartbeat_timeout = heartbeat_timeout
        self.buffer_size = buffer_size

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        self._last_frame_time = 0.0
        self._reconnect_count = 0
        self._total_frames = 0
        self._dropped_frames = 0
        self._fps = 0.0
        self._fps_history = []

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    def start(self):
        """启动采集线程"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop,
                                         daemon=True, name=f"stream-{self.name}")
        self._thread.start()
        logger.info(f"[{self.name}] Stream started: {self.source}")

    def stop(self):
        """停止采集"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        self._connected = False
        logger.info(f"[{self.name}] Stream stopped")

    def read(self) -> Optional[np.ndarray]:
        """读取最新帧（非阻塞）"""
        with self._frame_lock:
            if self._frame is not None:
                return self._frame.copy()
        return None

    def _capture_loop(self):
        """采集主循环"""
        while self._running:
            if not self._connected:
                if not self._connect():
                    time.sleep(self.reconnect_interval)
                    continue
            ret, frame = self._cap.read()
            if not ret or frame is None:
                self._handle_disconnect("read failed")
                continue
            now = time.time()
            with self._frame_lock:
                self._frame = frame
            self._last_frame_time = now
            self._total_frames += 1
            self._update_fps(now)
            if self._heartbeat_timeout > 0:
                if now - self._last_frame_time > self.heartbeat_timeout:
                    self._handle_disconnect("heartbeat timeout")

    def _connect(self) -> bool:
        """连接摄像头"""
        try:
            if self._cap:
                self._cap.release()
            cap = cv2.VideoCapture(self.source)
            if isinstance(self.source, str) and self.source.startswith("rtsp"):
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            if not cap.isOpened():
                cap.release()
                self._reconnect_count += 1
                if self.max_reconnect > 0 and self._reconnect_count > self.max_reconnect:
                    logger.error(f"[{self.name}] Max reconnect reached")
                    return False
                logger.warning(f"[{self.name}] Connect failed, retry {self._reconnect_count}")
                return False
            self._cap = cap
            self._connected = True
            self._reconnect_count = 0
            self._last_frame_time = time.time()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"[{self.name}] Connected: {w}x{h}@{fps:.0f}fps")
            return True
        except Exception as e:
            logger.error(f"[{self.name}] Connect error: {e}")
            return False

    def _handle_disconnect(self, reason: str):
        """处理断流"""
        self._connected = False
        self._reconnect_count += 1
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.warning(f"[{self.name}] Disconnected: {reason}, "
                        f"reconnect {self._reconnect_count}")

    def _update_fps(self, now: float):
        if self._last_frame_time > 0:
            dt = now - self._last_frame_time
            if dt > 0:
                self._fps_history.append(1.0 / dt)
                if len(self._fps_history) > 30:
                    self._fps_history.pop(0)
                self._fps = float(np.mean(self._fps_history))


class MultiStreamManager:
    """多路流管理器"""

    def __init__(self):
        self._streams: dict[str, CameraStream] = {}

    def add_stream(self, source, name: str, **kwargs) -> CameraStream:
        stream = CameraStream(source, name=name, **kwargs)
        self._streams[name] = stream
        return stream

    def start_all(self):
        for s in self._streams.values():
            s.start()

    def stop_all(self):
        for s in self._streams.values():
            s.stop()

    def get_frame(self, name: str) -> Optional[np.ndarray]:
        s = self._streams.get(name)
        return s.read() if s else None

    def get_all_frames(self) -> dict[str, Optional[np.ndarray]]:
        return {name: s.read() for name, s in self._streams.items()}

    def get_status(self) -> dict:
        return {
            name: {
                "connected": s.connected,
                "fps": round(s.fps, 1),
                "total_frames": s.total_frames,
                "dropped_frames": s.dropped_frames
            }
            for name, s in self._streams.items()
        }
