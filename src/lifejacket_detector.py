"""
人员救生衣穿戴识别模块
功能要求：依托船舶系缆监测相机，实时监测船舱外船员是否穿戴救生衣，
识别准确率不小于90%。
"""
import numpy as np
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class LifeJacketResult:
    """救生衣检测结果"""
    class_id: int          # 0=穿戴救生衣, 1=未穿戴救生衣
    class_name: str        # "lifejacket" or "no_lifejacket"
    confidence: float
    box: np.ndarray        # [x1, y1, x2, y2]
    has_lifejacket: bool


class LifeJacketDetector:
    """救生衣穿戴检测器（独立YOLO模型）"""

    CLASS_NAMES = {0: "lifejacket", 1: "no_lifejacket"}

    def __init__(self, config: dict):
        self.weights_path = config.get("weights", "weights/lifejacket_best.pt")
        self.conf_threshold = config.get("conf_threshold", 0.5)
        self.iou_threshold = config.get("iou_threshold", 0.45)
        self.device = config.get("device", "0")
        self.enabled = config.get("enabled", True)
        self._model = None

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.weights_path)
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model.predict(dummy, conf=0.1, verbose=False)

    def detect(self, image: np.ndarray,
               crew_boxes: Optional[List[np.ndarray]] = None) -> List[LifeJacketResult]:
        """
        检测救生衣穿戴情况。
        image: BGR帧
        crew_boxes: 如果提供船员检测框，只在船员区域内检测，提升效率和准确性
        """
        if not self.enabled:
            return []
        self._load_model()
        if crew_boxes:
            return self._detect_in_crew_regions(image, crew_boxes)
        return self._detect_full_frame(image)

    def _detect_full_frame(self, image: np.ndarray) -> List[LifeJacketResult]:
        results = self._model.predict(
            source=image, conf=self.conf_threshold,
            iou=self.iou_threshold, device=self.device,
            verbose=False, imgsz=640)
        return self._parse_results(results)

    def _detect_in_crew_regions(self, image: np.ndarray,
                                 crew_boxes: List[np.ndarray]) -> List[LifeJacketResult]:
        h, w = image.shape[:2]
        all_results = []
        for crew_box in crew_boxes:
            x1, y1, x2, y2 = crew_box[:4].astype(int)
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = image[y1:y2, x1:x2]
            results = self._model.predict(
                source=roi, conf=self.conf_threshold,
                iou=self.iou_threshold, device=self.device,
                verbose=False, imgsz=320)
            for r in self._parse_results(results):
                r.box[0] += x1
                r.box[1] += y1
                r.box[2] += x1
                r.box[3] += y1
                all_results.append(r)
        return all_results

    def _parse_results(self, results) -> List[LifeJacketResult]:
        out = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                for box, conf, cls_id in zip(boxes, confs, cls_ids):
                    name = self.CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                    out.append(LifeJacketResult(
                        class_id=cls_id, class_name=name,
                        confidence=float(conf), box=box,
                        has_lifejacket=(cls_id == 0)))
        return out

    def get_violations(self, results: List[LifeJacketResult]) -> List[LifeJacketResult]:
        """返回未穿戴救生衣的检测结果"""
        return [r for r in results if not r.has_lifejacket]
