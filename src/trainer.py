"""
YOLO11s训练模块
封装Ultralytics YOLO训练流程，支持数据增强、学习率调度、早停等。
"""
import os
from pathlib import Path
from loguru import logger


class Trainer:
    """YOLO11s训练器"""

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get("model_name", "yolo11s")
        self.data_yaml = config.get("data_yaml", "data/dataset.yaml")
        self.project = config.get("project", "runs/train")
        self.name = config.get("name", "mooring_yolo11s")

    def train(self,
              epochs: int = 100,
              imgsz: int = 640,
              batch: int = 16,
              lr0: float = 0.01,
              lrf: float = 0.01,
              warmup_epochs: int = 3,
              patience: int = 20,
              device: str = "0",
              workers: int = 8,
              resume: bool = False,
              pretrained_weights: str = "") -> str:
        """
        执行训练。
        返回: 最佳权重文件路径
        """
        from ultralytics import YOLO
        weights_path = f"{self.model_name}.pt"
        if pretrained_weights and os.path.exists(pretrained_weights):
            weights_path = pretrained_weights
        logger.info(f"Loading model: {weights_path}")
        model = YOLO(weights_path)
        logger.info(f"Starting training: {self.model_name}")
        logger.info(f"  epochs={epochs}, batch={batch}, imgsz={imgsz}")
        logger.info(f"  data={self.data_yaml}, device={device}")
        results = model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            lrf=lrf,
            warmup_epochs=warmup_epochs,
            patience=patience,
            device=device,
            workers=workers,
            project=self.project,
            name=self.name,
            exist_ok=True,
            resume=resume,
            # 数据增强
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=5.0,
            translate=0.1,
            scale=0.5,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            # 其他
            verbose=True,
            seed=42,
            deterministic=True,
        )
        best_path = os.path.join(self.project, self.name, "weights", "best.pt")
        if os.path.exists(best_path):
            logger.info(f"Best weights saved at: {best_path}")
        return best_path

    def validate(self, weights_path: str, device: str = "0") -> dict:
        """
        在验证集上评估模型性能。
        返回: 评估指标字典
        """
        from ultralytics import YOLO
        model = YOLO(weights_path)
        results = model.val(
            data=self.data_yaml,
            device=device,
            verbose=True
        )
        metrics = {
            "mAP50": float(results.box.map50),
            "mAP50_95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
            "per_class": {}
        }
        names = results.names
        for cls_id, ap in enumerate(results.box.ap50):
            cls_name = names.get(cls_id, f"class_{cls_id}")
            metrics["per_class"][cls_name] = float(ap)
        logger.info(f"Validation results: {metrics}")
        return metrics
