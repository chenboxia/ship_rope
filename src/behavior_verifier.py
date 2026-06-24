"""
船员带缆行为验证模块
判断船员是否正在执行系缆操作。
"""
import numpy as np
from typing import List, Tuple
from src.detector import Detection
from src.geometry import box_center, point_distance, compute_iou


class BehaviorVerifier:
    """船员带缆行为验证器"""

    def __init__(self, config: dict):
        self.proximity_threshold = config.get("proximity_threshold", 120)
        self.iou_threshold = config.get("iou_threshold", 0.05)

    def verify(self, crew_detections: List[Detection],
               rope_detections: List[Detection],
               anchor_points: List[Tuple[int, int]]) -> List[dict]:
        """
        验证每个船员是否正在带缆操作。
        crew_detections: 船员检测列表
        rope_detections: 缆绳检测列表
        anchor_points: 各缆绳的系结锚点列表
        返回: 每个船员的操作状态列表
        """
        results = []
        for crew in crew_detections:
            crew_center = box_center(crew.box)
            is_near_anchor = False
            nearest_anchor_dist = float("inf")
            for anchor in anchor_points:
                if anchor is None:
                    continue
                d = point_distance(crew_center, anchor)
                if d < nearest_anchor_dist:
                    nearest_anchor_dist = d
                if d < self.proximity_threshold:
                    is_near_anchor = True

            is_touching_rope = False
            max_rope_iou = 0.0
            for rope in rope_detections:
                iou = compute_iou(crew.box, rope.box)
                if iou > max_rope_iou:
                    max_rope_iou = iou
                if iou >= self.iou_threshold:
                    is_touching_rope = True

            is_active = is_near_anchor and is_touching_rope
            results.append({
                "detection": crew,
                "is_near_anchor": is_near_anchor,
                "nearest_anchor_dist": nearest_anchor_dist,
                "is_touching_rope": is_touching_rope,
                "max_rope_iou": max_rope_iou,
                "is_active": is_active
            })
        return results
