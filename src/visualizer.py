"""
可视化模块
负责在视频帧上绘制检测结果、状态标注、壁面区域等。
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from src.detector import Detection
from src.rope_analyzer import RopeState
from src.state_machine import MooringState


class Visualizer:
    """可视化绘制器"""

    def __init__(self, config: dict):
        colors = config.get("colors", {})
        self.color_moored = tuple(colors.get("moored", [0, 255, 0]))
        self.color_unmoored = tuple(colors.get("unmoored", [0, 0, 255]))
        self.color_switching = tuple(colors.get("switching", [0, 255, 255]))
        self.color_crew_active = tuple(colors.get("crew_active", [0, 255, 255]))
        self.color_wall = tuple(colors.get("wall_region", [255, 200, 0]))
        self.thickness = config.get("thickness", 2)
        self.font_scale = config.get("font_scale", 0.6)

    def draw(self, frame: np.ndarray,
             rope_states: List[RopeState],
             rope_sm_states: List[MooringState],
             crew_activities: List[dict],
             wall_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        在帧上绘制所有标注。
        """
        vis = frame.copy()
        if wall_mask is not None:
            vis = self._draw_wall_region(vis, wall_mask)
        for i, (rope_state, sm_state) in enumerate(zip(rope_states, rope_sm_states)):
            vis = self._draw_rope_state(vis, rope_state, sm_state, i)
        for activity in crew_activities:
            vis = self._draw_crew_activity(vis, activity)
        return vis

    def _draw_wall_region(self, frame: np.ndarray,
                           wall_mask: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, self.color_wall, 1)
        alpha = 0.1
        mask_bool = wall_mask > 0
        overlay[mask_bool] = (
            (1 - alpha) * overlay[mask_bool] + alpha * np.array(self.color_wall)
        ).astype(np.uint8)
        return overlay

    def _draw_rope_state(self, frame: np.ndarray,
                          rope_state: RopeState,
                          sm_state: MooringState,
                          index: int) -> np.ndarray:
        if sm_state == MooringState.MOORED:
            color = self.color_moored
            label = "MOORED"
        elif sm_state == MooringState.SWITCHING:
            color = self.color_switching
            label = "SWITCHING"
        elif sm_state == MooringState.MONITORING:
            color = self.color_switching
            label = "MONITORING"
        else:
            color = self.color_unmoored
            label = "UNMOORED"

        box = rope_state.rope_detection.box[:4].astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]),
                       color, self.thickness)

        # 端点标记
        if rope_state.anchor_point:
            cv2.circle(frame, rope_state.anchor_point, 5, color, -1)
        if rope_state.ship_point:
            cv2.circle(frame, rope_state.ship_point, 5, color, -1)

        # 端点连线
        if rope_state.anchor_point and rope_state.ship_point:
            cv2.line(frame, rope_state.anchor_point, rope_state.ship_point,
                      color, 1, cv2.LINE_AA)

        # 标签文字
        text = f"Rope{index}: {label} T:{rope_state.tension_ratio:.2f}"
        y_pos = max(box[1] - 8, 15)
        cv2.putText(frame, text, (box[0], y_pos),
                     cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                     color, self.thickness, cv2.LINE_AA)
        return frame

    def _draw_crew_activity(self, frame: np.ndarray,
                             activity: dict) -> np.ndarray:
        det = activity["detection"]
        box = det.box[:4].astype(int)
        if activity["is_active"]:
            color = self.color_crew_active
            label = "ACTIVE"
            style = cv2.LINE_4  # 虚线效果
        else:
            color = (200, 200, 200)
            label = "IDLE"
            style = cv2.LINE_8

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]),
                       color, self.thickness, style)
        y_pos = max(box[1] - 5, 15)
        cv2.putText(frame, label, (box[0], y_pos),
                     cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.8,
                     color, 1, cv2.LINE_AA)
        return frame

    def draw_status_bar(self, frame: np.ndarray,
                         rope_sm_states: List[MooringState]) -> np.ndarray:
        """在帧顶部绘制状态汇总栏"""
        h, w = frame.shape[:2]
        bar_h = 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (40, 40, 40), -1)
        frame[:bar_h] = cv2.addWeighted(overlay[:bar_h], 0.7,
                                         frame[:bar_h], 0.3, 0)
        moored = sum(1 for s in rope_sm_states if s == MooringState.MOORED)
        unmoored = sum(1 for s in rope_sm_states if s == MooringState.UNMOORED)
        switching = sum(1 for s in rope_sm_states
                        if s in (MooringState.SWITCHING, MooringState.MONITORING))
        text = (f"Ropes: {len(rope_sm_states)}  "
                f"Moored: {moored}  Unmoored: {unmoored}  "
                f"Switching: {switching}")
        cv2.putText(frame, text, (10, 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                     (255, 255, 255), 1, cv2.LINE_AA)
        return frame
