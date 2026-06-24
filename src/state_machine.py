"""
时间规则防误报状态机模块
实现五阶段状态判定：系结确认、脱系触发、船员辅助判断、超时报警、换缆完成检测。
"""
import time
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field


class MooringState(Enum):
    """系缆状态枚举"""
    MOORED = "moored"          # 已系结
    UNMOORED = "unmoored"      # 未系结（异常）
    SWITCHING = "switching"    # 换缆中
    MONITORING = "monitoring"  # 监控中（脱系但未判定）


@dataclass
class RopeStateMachine:
    """单根缆绳的状态机"""
    rope_id: int
    confirm_frames: int = 3
    timeout_seconds: float = 30.0

    # 内部状态
    state: MooringState = MooringState.UNMOORED
    moored_frame_count: int = 0          # 连续系结帧数
    unmoored_frame_count: int = 0        # 连续脱系帧数
    disconnect_start_time: Optional[float] = None
    crew_active: bool = False
    alarm_triggered: bool = False
    last_update_time: float = field(default_factory=time.time)

    def update(self, is_moored: bool, is_crew_active: bool) -> MooringState:
        """
        根据当前帧的检测结果更新状态机。
        is_moored: 当前帧缆绳是否满足系结判定
        is_crew_active: 当前帧系结区域附近是否有活跃船员
        返回: 更新后的状态
        """
        now = time.time()
        self.crew_active = is_crew_active
        self.last_update_time = now

        if is_moored:
            return self._handle_moored()
        else:
            return self._handle_unmoored(now)

    def _handle_moored(self) -> MooringState:
        """处理系结状态"""
        self.moored_frame_count += 1
        self.unmoored_frame_count = 0
        if self.moored_frame_count >= self.confirm_frames:
            # 确认系结，重置所有脱系状态
            old_state = self.state
            self.state = MooringState.MOORED
            self.disconnect_start_time = None
            self.alarm_triggered = False
            return self.state
        # 帧数不够，维持当前状态
        return self.state

    def _handle_unmoored(self, now: float) -> MooringState:
        """处理脱系状态"""
        self.unmoored_frame_count += 1
        self.moored_frame_count = 0

        if self.unmoored_frame_count < self.confirm_frames:
            # 帧数不够确认，维持当前状态
            return self.state

        # 确认脱系，启动状态机逻辑
        if self.state == MooringState.MOORED:
            # 从系结状态转入脱系，启动计时
            self.state = MooringState.MONITORING
            self.disconnect_start_time = now
            self.alarm_triggered = False
            return self.state

        if self.state == MooringState.MONITORING:
            elapsed = now - self.disconnect_start_time if self.disconnect_start_time else 0
            if self.crew_active:
                # 有船员活动，判定为换缆中
                self.state = MooringState.SWITCHING
                return self.state
            elif elapsed >= self.timeout_seconds:
                # 超时无船员活动，触发报警
                self.state = MooringState.UNMOORED
                self.alarm_triggered = True
                return self.state
            else:
                # 继续监控
                return self.state

        if self.state == MooringState.SWITCHING:
            elapsed = now - self.disconnect_start_time if self.disconnect_start_time else 0
            if not self.crew_active:
                # 船员离开，转入监控
                self.state = MooringState.MONITORING
                return self.state
            elif elapsed >= self.timeout_seconds:
                # 超时，即使有船员也报警
                self.state = MooringState.UNMOORED
                self.alarm_triggered = True
                return self.state
            else:
                return self.state

        if self.state == MooringState.UNMOORED:
            # 已经报警，等待系结恢复
            return self.state

        return self.state

    def reset(self):
        """重置状态机"""
        self.state = MooringState.UNMOORED
        self.moored_frame_count = 0
        self.unmoored_frame_count = 0
        self.disconnect_start_time = None
        self.crew_active = False
        self.alarm_triggered = False
