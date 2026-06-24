"""
主监测流水线
4类模型：ship / rope / person / lifejacket
人与救生衣通过IoU关联判定穿戴状态。
"""
import time
import numpy as np
from typing import Dict
from loguru import logger
from src.detector import Detector, Detection
from src.rope_analyzer import RopeAnalyzer, WallConfig
from src.behavior_verifier import BehaviorVerifier
from src.state_machine import RopeStateMachine, MooringState
from src.data_exporter import DataExporter
from src.visualizer import Visualizer


class MonitorPipeline:
    def __init__(self, config):
        self.config = config
        mc = config["model"]
        self.detector = Detector(
            weights_path=mc["weights"], conf=mc["conf_threshold"],
            iou=mc["iou_threshold"], device=mc["device"])
        self.rope_analyzer = RopeAnalyzer(config["rope_analysis"])
        self.behavior_verifier = BehaviorVerifier(config["crew_behavior"])
        sm_cfg = config["state_machine"]
        self.state_machines: Dict[int, RopeStateMachine] = {}
        self.confirm_frames = sm_cfg["confirm_frames"]
        self.timeout_seconds = sm_cfg["timeout_seconds"]
        self.lj_iou_threshold = config.get("lifejacket", {}).get("iou_threshold", 0.1)
        wc = config["wall_region"]
        self.wall_config = WallConfig(
            points=wc.get("points"), default_ratio=wc.get("default_ratio", 0.15))
        self.visualizer = Visualizer(config["visualization"])
        self.data_exporter = DataExporter(config.get("export", {}))
        self.frame_count = 0

    def process_frame(self, frame):
        self.frame_count += 1
        ts = time.time()
        result = {
            "frame_id": self.frame_count, "timestamp": ts, "frame": frame,
            "rope_states": [], "crew_activities": [], "sm_states": [],
            "lifejacket_wearing": [], "lifejacket_violations": [],
            "alarms": [], "annotated_frame": frame
        }
        try:
            # 1. 一次推理，4类目标
            detections = self.detector.predict(frame)
            ships = self.detector.filter_by_class(detections, "ship")
            ropes = self.detector.filter_by_class(detections, "rope")
            persons = self.detector.filter_by_class(detections, "person")
            lj_dets = self.detector.filter_by_class(detections, "lifejacket")

            # 2. 人与救生衣IoU关联
            wearing, not_wearing = self.detector.associate_lifejackets(
                persons, lj_dets, self.lj_iou_threshold)
            result["lifejacket_wearing"] = wearing
            result["lifejacket_violations"] = not_wearing

            # 3. 系缆形态分析
            rope_states = self.rope_analyzer.analyze(frame, ropes, ships, self.wall_config)
            result["rope_states"] = rope_states

            # 4. 船员行为验证（所有人参与，不管穿没穿救生衣）
            anchors = [rs.anchor_point for rs in rope_states]
            crew_activities = self.behavior_verifier.verify(persons, ropes, anchors)
            result["crew_activities"] = crew_activities

            # 5. 状态机
            any_active = any(a["is_active"] for a in crew_activities)
            sm_states, alarms = self._update_sm(rope_states, any_active)

            # 6. 救生衣报警
            for p in not_wearing:
                alarms.append({"type": "lifejacket",
                               "message": "No lifejacket (conf=%.2f)" % p.confidence})
            result["sm_states"] = sm_states
            result["alarms"] = alarms

            # 7. 可视化
            wm = self.rope_analyzer._build_wall_mask(
                self.wall_config, frame.shape[1], frame.shape[0])
            vis = self.visualizer.draw(frame, rope_states, sm_states, crew_activities, wm)
            vis = self._draw_lifejacket(vis, wearing, not_wearing)
            vis = self.visualizer.draw_status_bar(vis, sm_states)
            result["annotated_frame"] = vis

            # 8. 导出
            self.data_exporter.export_frame_result(
                self.frame_count, ts, rope_states, sm_states,
                crew_activities, None, not_wearing, alarms)
        except Exception as e:
            logger.error("Frame %d error: %s", self.frame_count, e)
        return result

    def _update_sm(self, rope_states, any_active):
        sm_states, alarms = [], []
        for i, rs in enumerate(rope_states):
            if i not in self.state_machines:
                self.state_machines[i] = RopeStateMachine(
                    i, self.confirm_frames, self.timeout_seconds)
            sm = self.state_machines[i]
            st = sm.update(rs.is_moored, any_active)
            sm_states.append(st)
            if sm.alarm_triggered:
                alarms.append({"type": "mooring", "rope_id": i, "state": st.value,
                               "message": "Rope %d: %s" % (i, st.value.upper())})
        return sm_states, alarms

    def _draw_lifejacket(self, frame, wearing, not_wearing):
        import cv2
        vis = frame.copy()
        for person, lj in wearing:
            pb = person.box[:4].astype(int)
            cv2.rectangle(vis, (pb[0], pb[1]), (pb[2], pb[3]), (0, 255, 0), 1)
        for person in not_wearing:
            pb = person.box[:4].astype(int)
            cv2.rectangle(vis, (pb[0], pb[1]), (pb[2], pb[3]), (0, 0, 255), 2)
            cv2.putText(vis, "NO LJ", (pb[0], pb[1] - 5),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return vis

    def reset(self):
        self.state_machines.clear()
        self.frame_count = 0
