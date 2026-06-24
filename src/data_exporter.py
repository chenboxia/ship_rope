"""
结构化数据输出模块
将监测结果输出为结构化JSON，对接船舶过闸综合辅助终端。
"""
import json
import time
import os
import numpy as np
from datetime import datetime
from typing import List, Optional


class NumpyEncoder(json.JSONEncoder):
    """处理numpy类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class DataExporter:
    """结构化数据导出器"""

    def __init__(self, config: dict):
        self.output_dir = config.get("output_dir", "outputs/")
        self.export_json = config.get("export_json", True)
        self.export_interval = config.get("export_interval", 1.0)
        self._last_export_time = 0.0
        os.makedirs(self.output_dir, exist_ok=True)

    def export_frame_result(self, frame_id, timestamp,
                             rope_states, sm_states,
                             crew_activities,
                             distance_alerts=None,
                             lifejacket_violations=None,
                             alarms=None):
        now = time.time()
        if now - self._last_export_time < self.export_interval:
            return None
        self._last_export_time = now
        record = {
            "frame_id": int(frame_id),
            "timestamp": float(timestamp),
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "mooring_status": self._build_mooring_status(rope_states, sm_states),
            "crew_status": self._build_crew_status(crew_activities),
            "distance_status": self._build_distance_status(distance_alerts),
            "lifejacket_status": self._build_lifejacket_status(lifejacket_violations),
            "alarms": alarms or []
        }
        if self.export_json:
            self._write_json(record)
        return record

    def _build_mooring_status(self, rope_states, sm_states):
        out = []
        for i, (rs, sm) in enumerate(zip(rope_states, sm_states)):
            entry = {
                "rope_id": i,
                "state": sm.value if hasattr(sm, 'value') else str(sm),
                "is_moored": bool(rs.is_moored),
                "tension_ratio": round(float(rs.tension_ratio), 3),
                "angle_deg": round(float(rs.angle_to_wall_normal), 1),
                "ship_endpoint_dist_px": round(float(rs.ship_endpoint_dist), 1),
                "confidence": round(float(rs.confidence), 3),
            }
            if rs.anchor_point:
                entry["anchor_point"] = [int(x) for x in rs.anchor_point]
            if rs.ship_point:
                entry["ship_point"] = [int(x) for x in rs.ship_point]
            out.append(entry)
        return out

    def _build_crew_status(self, crew_activities):
        out = []
        for i, act in enumerate(crew_activities):
            det = act["detection"]
            out.append({
                "crew_id": i,
                "is_active": bool(act["is_active"]),
                "near_anchor": bool(act["is_near_anchor"]),
                "touching_rope": bool(act["is_touching_rope"]),
                "confidence": round(float(det.confidence), 3)
            })
        return out

    def _build_distance_status(self, distance_alerts):
        if not distance_alerts:
            return []
        out = []
        for i, alert in enumerate(distance_alerts):
            out.append({
                "ship_id": i,
                "distance_m": round(float(alert.min_distance_meters), 2),
                "threshold_m": float(alert.warning_threshold_m),
                "is_warning": bool(alert.is_warning)
            })
        return out

    def _build_lifejacket_status(self, violations):
        if not violations:
            return {"violations": 0, "details": []}
        details = []
        for v in violations:
            details.append({
                "confidence": round(float(v.confidence), 3),
                "box": [int(x) for x in v.box[:4]]
            })
        return {"violations": len(violations), "details": details}

    def _write_json(self, record):
        ts = datetime.fromtimestamp(record["timestamp"]).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"frame_{ts}_{record['frame_id']:06d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
