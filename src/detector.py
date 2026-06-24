"""
YOLO11s目标检测模块
4类：ship / rope / person / lifejacket
人是人，救生衣是救生衣，穿戴关系通过IoU关联判定。
"""
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: np.ndarray  # [x1, y1, x2, y2]


CLASS_NAMES = {
    0: "ship",
    1: "rope",
    2: "person",
    3: "lifejacket"
}


def compute_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Detector:
    def __init__(self, weights_path, conf=0.5, iou=0.45, device="0"):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.conf = conf
        self.iou = iou
        self.device = device
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.predict(dummy)

    def predict(self, image):
        results = self.model.predict(
            source=image, conf=self.conf, iou=self.iou,
            device=self.device, verbose=False, imgsz=640)
        detections = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                for box, conf, cid in zip(boxes, confs, cls_ids):
                    name = CLASS_NAMES.get(cid, "class_%d" % cid)
                    detections.append(Detection(cid, name, float(conf), box))
        return detections

    def filter_by_class(self, detections, class_name):
        return [d for d in detections if d.class_name == class_name]

    def associate_lifejackets(self, person_dets, lj_dets,
                               iou_threshold=0.1):
        """
        关联人与救生衣，判定穿戴状态。
        person_dets: person检测列表
        lj_dets: lifejacket检测列表
        返回: (wearing_list, not_wearing_list)
          wearing_list: (person_det, lifejacket_det) 元组列表
          not_wearing_list: 未穿戴救生衣的person_det列表
        """
        wearing = []
        not_wearing = []
        lj_matched = set()
        for person in person_dets:
            best_iou = 0.0
            best_lj = None
            for j, lj in enumerate(lj_dets):
                if j in lj_matched:
                    continue
                iou = compute_iou(person.box, lj.box)
                if iou > best_iou:
                    best_iou = iou
                    best_lj = (j, lj)
            if best_iou >= iou_threshold and best_lj is not None:
                wearing.append((person, best_lj[1]))
                lj_matched.add(best_lj[0])
            else:
                not_wearing.append(person)
        return wearing, not_wearing
