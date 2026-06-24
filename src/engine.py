"""
工业级引擎模块
主进程编排、watchdog、GPU监控、配置热更新、优雅停机。
"""
import os
import sys
import time
import signal
import threading
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger
from src.stream_manager import MultiStreamManager
from src.monitor import MonitorPipeline
from src.model_optimizer import get_optimal_weights
from src.api_server import APIServer


class GPUMonitor:
    """GPU资源监控"""
    def __init__(self, mem_warn_ratio=0.85):
        self.mem_warn_ratio = mem_warn_ratio
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def get_status(self):
        try:
            import torch
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(0).total_mem / 1024**3
                ratio = alloc / total if total > 0 else 0
                return {"gpu_available": True, "mem_total_gb": round(total, 2),
                        "mem_allocated_gb": round(alloc, 2), "mem_usage_ratio": round(ratio, 3)}
        except Exception:
            pass
        return {"gpu_available": False, "mem_usage_ratio": 0}

    def _loop(self):
        while self._running:
            st = self.get_status()
            if st.get("mem_usage_ratio", 0) > self.mem_warn_ratio:
                logger.warning("GPU memory high: %.1f%%", st["mem_usage_ratio"] * 100)
                try:
                    import torch, gc; gc.collect(); torch.cuda.empty_cache()
                except Exception:
                    pass
            time.sleep(10)


class ConfigWatcher:
    """配置文件热更新"""
    def __init__(self, config_path, callback=None):
        self.config_path = config_path
        self.callback = callback
        self._last_mtime = 0.0
        self._running = False
        self._config = {}

    def load(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)
        self._last_mtime = os.path.getmtime(self.config_path)
        return self._config

    @property
    def config(self):
        return self._config

    def start(self):
        self._running = True
        threading.Thread(target=self._watch_loop, daemon=True).start()

    def stop(self):
        self._running = False

    def _watch_loop(self):
        while self._running:
            time.sleep(5)
            try:
                mtime = os.path.getmtime(self.config_path)
                if mtime > self._last_mtime:
                    new_cfg = self.load()
                    logger.info("Config hot-reloaded")
                    if self.callback:
                        self.callback(new_cfg)
            except Exception as e:
                logger.error("Config watch error: %s", e)


class Engine:
    """工业级监测引擎"""
    def __init__(self, config_path="configs/config.yaml"):
        self.config_path = config_path
        self._shutdown_event = threading.Event()
        self._streams = MultiStreamManager()
        self._gpu_monitor = GPUMonitor()
        self._config_watcher = ConfigWatcher(config_path)
        self._pipeline = None
        self._api_server = None
        self._stats = {"start_time": 0, "total_frames": 0,
                       "total_alarms": 0, "errors": 0}

    def run(self, sources, show=False):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        config = self._config_watcher.load()
        self._setup_logging(config)
        logger.info("=" * 60)
        logger.info("Mooring Monitor Engine Starting")
        logger.info("=" * 60)
        self._gpu_monitor.start()
        self._config_watcher.start()
        self._pipeline = MonitorPipeline(config)
        for i, src in enumerate(sources):
            self._streams.add_stream(src, "cam%d" % i,
                                      reconnect_interval=5.0, heartbeat_timeout=15.0)
        self._streams.start_all()
        api_cfg = config.get("api", {})
        if api_cfg.get("enabled", False):
            self._api_server = APIServer(api_cfg)
            self._api_server.start()
        time.sleep(2)
        self._stats["start_time"] = time.time()
        logger.info("Engine ready, entering main loop")
        try:
            self._main_loop(show)
        except Exception as e:
            logger.error("Engine fatal error: %s", e)
        finally:
            self._shutdown()

    def _main_loop(self, show):
        import cv2
        while not self._shutdown_event.is_set():
            frames = self._streams.get_all_frames()
            for name, frame in frames.items():
                if frame is None:
                    continue
                try:
                    result = self._pipeline.process_frame(frame)
                    self._stats["total_frames"] += 1
                    if result["alarms"]:
                        self._stats["total_alarms"] += len(result["alarms"])
                        for alarm in result["alarms"]:
                            logger.warning("ALARM [%s]: %s", name, alarm["message"])
                    if self._api_server:
                        rec = {"stream": name, "frame_id": self._stats["total_frames"],
                               "timestamp": time.time(),
                               "mooring": [{"state": s.value, "moored": rs.is_moored}
                                            for rs, s in zip(result["rope_states"], result["sm_states"])],
                               "alarms": result["alarms"]}
                        APIServer.push_record(rec)
                    if show:
                        cv2.imshow(name, result["annotated_frame"])
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            self._shutdown_event.set()
                            break
                except Exception as e:
                    self._stats["errors"] += 1
                    logger.error("Process error [%s]: %s", name, e)
            if self._stats["total_frames"] % 500 == 0 and self._stats["total_frames"] > 0:
                self._log_stats()
            if not any(f is not None for f in frames.values()):
                time.sleep(0.01)

    def _log_stats(self):
        elapsed = time.time() - self._stats["start_time"]
        fps = self._stats["total_frames"] / elapsed if elapsed > 0 else 0
        gpu = self._gpu_monitor.get_status()
        gpu_mem = gpu.get("mem_usage_ratio", 0)
        logger.info("Stats: frames=%d, fps=%.1f, alarms=%d, errors=%d, gpu=%.1f%%",
                     self._stats["total_frames"], fps,
                     self._stats["total_alarms"], self._stats["errors"],
                     gpu_mem * 100)
        for name, st in self._streams.get_status().items():
            logger.info("  %s: connected=%s, fps=%.1f", name, st["connected"], st["fps"])

    def _signal_handler(self, sig, frame):
        logger.info("Signal %d received, shutting down", sig)
        self._shutdown_event.set()

    def _shutdown(self):
        logger.info("Shutting down engine...")
        self._streams.stop_all()
        if self._api_server:
            self._api_server.stop()
        self._gpu_monitor.stop()
        self._config_watcher.stop()
        self._log_stats()
        logger.info("Engine stopped")

    def _setup_logging(self, config):
        log_cfg = config.get("logging", {})
        log_file = log_cfg.get("file", "outputs/monitor.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logger.add(log_file, rotation=log_cfg.get("rotation", "10 MB"),
                    level=log_cfg.get("level", "INFO"), encoding="utf-8")
