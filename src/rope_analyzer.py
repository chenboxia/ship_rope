"""
缆绳形态分析模块
主判据：绳子真实端点是否在壁面区域内（基于壁面标定）
辅助判据：端点稳定性、张紧度（置信度加分+兜底）
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from src.detector import Detection
from src.geometry import (
    compute_rope_contour, compute_tension_ratio,
    compute_rope_angle_at_endpoint, extract_endpoints_from_roi,
    point_distance, point_to_box_distance
)


@dataclass
class WallConfig:
    points: Optional[np.ndarray] = None
    default_ratio: float = 0.15
    wall_normal: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    image_width: int = 0
    image_height: int = 0

    def __post_init__(self):
        if self.points is not None and len(self.points) > 0:
            self.points = np.array(self.points, dtype=np.int32)


@dataclass
class RopeState:
    rope_detection: Detection
    anchor_point: Optional[Tuple[int, int]] = None
    ship_point: Optional[Tuple[int, int]] = None
    anchor_on_wall: bool = False     # 主判据：绳子端点是否在壁面上
    tension_ratio: float = 0.0
    angle_to_wall_normal: float = 90.0
    anchor_stable: bool = False
    is_moored: bool = False
    is_slack: bool = False
    confidence: float = 0.0


class RopeAnalyzer:
    def __init__(self, config: dict):
        self.tension_threshold = config.get("tension_threshold", 0.3)
        self.min_rope_length = config.get("min_rope_length", 20)
        self.anchor_stability_threshold = config.get("anchor_stability_threshold", 30)
        self._anchor_history: dict = {}

    def analyze(self, image, rope_detections, ship_detections, wall_config):
        h, w = image.shape[:2]
        if wall_config.image_width == 0:
            wall_config.image_width = w
            wall_config.image_height = h
        wall_mask = self._build_wall_mask(wall_config, w, h)
        states = []
        for i, rope_det in enumerate(rope_detections):
            state = self._analyze_single_rope(
                image, rope_det, wall_mask, wall_config, i)
            states.append(state)
        return states

    def _analyze_single_rope(self, image, rope_det, wall_mask, wall_config, rope_idx):
        state = RopeState(rope_detection=rope_det)
        box = rope_det.box
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box[:4].astype(int)

        # ====== 第一步：从框内提取绳子真实端点 ======
        rope_mask = compute_rope_contour(image, box, method="edge")
        endpoints = None
        if rope_mask is not None and np.sum(rope_mask) > 0:
            endpoints = extract_endpoints_from_roi(
                rope_mask, offset_x=max(0, x1), offset_y=max(0, y1))
        if endpoints is None:
            endpoints = self._endpoints_from_box_pca(image, box)
        if endpoints is None:
            # 兜底用框的对角线
            endpoints = ((int(x1), int(y1)), (int(x2), int(y2)))
        p1, p2 = endpoints

        # ====== 第二步：判断哪个端点靠近壁面 ======
        wall_box = np.array([0, 0, w * wall_config.default_ratio, h])
        if point_to_box_distance(p1, wall_box) <= point_to_box_distance(p2, wall_box):
            anchor, ship_end = p1, p2
        else:
            anchor, ship_end = p2, p1
        state.anchor_point = anchor
        state.ship_point = ship_end

        # 用壁面掩码校正
        if wall_mask is not None and np.sum(wall_mask) > 0:
            ax, ay = int(anchor[0]), int(anchor[1])
            if 0 <= ax < wall_mask.shape[1] and 0 <= ay < wall_mask.shape[0]:
                if wall_mask[ay, ax] == 0:
                    state.anchor_point, state.ship_point = ship_end, anchor

        # ====== 第三步：主判据——绳子端点是否在壁面区域内 ======
        state.anchor_on_wall = self._is_point_on_wall(state.anchor_point, wall_mask)

        # 长度检查
        rope_length = point_distance(state.anchor_point, state.ship_point)
        if rope_length < self.min_rope_length:
            return state

        # ====== 辅助指标 ======
        if rope_mask is not None:
            state.tension_ratio = compute_tension_ratio(
                state.anchor_point, state.ship_point, rope_mask,
                offset_x=max(0, x1), offset_y=max(0, y1))
        state.angle_to_wall_normal = compute_rope_angle_at_endpoint(
            state.anchor_point, state.ship_point, wall_config.wall_normal)
        state.anchor_stable = self._check_anchor_stability(rope_idx, state.anchor_point)

        # ====== 综合判定 ======
        if state.anchor_on_wall:
            # 端点在墙上 = 系着
            state.is_moored = True
        elif state.anchor_stable:
            # 端点不在墙上但连续稳定（可能标定偏了或边缘情况），兜底
            state.is_moored = True
        else:
            state.is_moored = False

        # 松缆
        if state.is_moored and state.tension_ratio < self.tension_threshold:
            state.is_slack = True

        # 置信度
        score = 0.0
        if state.anchor_on_wall:
            score += 0.7
        if state.anchor_stable:
            score += 0.2
        if state.tension_ratio >= self.tension_threshold:
            score += 0.1
        state.confidence = min(score, 1.0)
        return state

    def _is_point_on_wall(self, point, wall_mask):
        """绳子端点是否在壁面标定区域内"""
        if point is None or wall_mask is None:
            return False
        px, py = int(point[0]), int(point[1])
        h, w = wall_mask.shape
        if 0 <= px < w and 0 <= py < h:
            return wall_mask[py, px] > 0
        return False

    def _check_anchor_stability(self, rope_idx, current_anchor):
        if current_anchor is None:
            return False
        history = self._anchor_history.get(rope_idx, [])
        history.append(current_anchor)
        if len(history) > 10:
            history = history[-10:]
        self._anchor_history[rope_idx] = history
        if len(history) < 3:
            return False
        recent = history[-5:]
        max_dist = 0
        for i in range(1, len(recent)):
            d = point_distance(recent[i], recent[i-1])
            if d > max_dist:
                max_dist = d
        return max_dist <= self.anchor_stability_threshold

    def _endpoints_from_box_pca(self, image, box):
        import cv2
        x1, y1, x2, y2 = box[:4].astype(int)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        roi = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pts = np.column_stack(np.where(thresh > 0))
        if len(pts) < 2:
            return None
        pts_xy = pts[:, ::-1].astype(float)
        if len(pts_xy) < 3:
            return ((int(pts_xy[0][0]) + x1, int(pts_xy[0][1]) + y1),
                    (int(pts_xy[-1][0]) + x1, int(pts_xy[-1][1]) + y1))
        mean = np.mean(pts_xy, axis=0)
        centered = pts_xy - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal = eigenvectors[:, np.argmax(eigenvalues)]
        projections = centered @ principal
        idx_min = np.argmin(projections)
        idx_max = np.argmax(projections)
        p1 = (int(pts_xy[idx_min][0]) + x1, int(pts_xy[idx_min][1]) + y1)
        p2 = (int(pts_xy[idx_max][0]) + x1, int(pts_xy[idx_max][1]) + y1)
        return (p1, p2)

    def _build_wall_mask(self, config, w, h):
        import cv2
        mask = np.zeros((h, w), dtype=np.uint8)
        if config.points is not None and len(config.points) >= 3:
            cv2.fillPoly(mask, [config.points], 255)
        else:
            mask[:, :int(w * config.default_ratio)] = 255
        return mask
