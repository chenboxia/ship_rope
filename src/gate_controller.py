"""
闸门状态控制模块
闸门开启期间暂停系缆报警，闸门关闭后恢复监测。
支持网络接口接收闸门状态，也支持手动切换。
"""
import time
import threading
from enum import Enum
from loguru import logger


class GateState(Enum):
    CLOSED = "closed"    # 闸门关闭，正常监测
    OPENING = "opening"  # 闸门开启中，暂停报警
    OPEN = "open"        # 闸门已开，暂停报警
    CLOSING = "closing"  # 闸门关闭中，准备恢复


class GateController:
    def __init__(self, config: dict):
        self.enabled = config.get("enabled", False)
        self.state = GateState.CLOSED
        self._lock = threading.Lock()
        self._last_update = time.time()
        # 闸门开启后多久恢复监测（秒），防止刚关上就报警
        self.resume_delay = config.get("resume_delay", 60)
        self._close_time = 0.0

    @property
    def should_monitor(self) -> bool:
        """当前是否应该监测系缆状态"""
        if not self.enabled:
            return True
        with self._lock:
            if self.state in (GateState.OPENING, GateState.OPEN):
                return False
            if self.state == GateState.CLOSING:
                if time.time() - self._close_time < self.resume_delay:
                    return False
            return True

    def set_gate_state(self, state_str: str):
        """外部设置闸门状态，state_str: "open" 或 "closed" """
        with self._lock:
            old = self.state
            if state_str == "open":
                self.state = GateState.OPEN
            elif state_str == "closed":
                if old in (GateState.OPEN, GateState.OPENING):
                    self.state = GateState.CLOSING
                    self._close_time = time.time()
                    logger.info("Gate closed, resuming monitoring in %ds",
                                self.resume_delay)
                else:
                    self.state = GateState.CLOSED
            self._last_update = time.time()
            if old != self.state:
                logger.info("Gate state: %s -> %s", old.value, self.state.value)

    def get_status(self) -> dict:
        return {
            "enabled": self.enabled,
            "state": self.state.value,
            "should_monitor": self.should_monitor,
            "last_update": self._last_update
        }
