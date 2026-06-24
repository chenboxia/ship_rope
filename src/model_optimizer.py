"""
模型优化模块
支持ONNX导出、TensorRT INT8量化，适配100 TOPS INT8硬件。
"""
import os
import time
import numpy as np
from pathlib import Path
from loguru import logger


class ModelOptimizer:
    """模型优化器"""

    def __init__(self, model_path: str, device: str = "0"):
        self.model_path = model_path
        self.device = device
        self._model = None

    def export_onnx(self, output_path: str = None, imgsz: int = 640,
                     simplify: bool = True, opset: int = 17) -> str:
        """导出ONNX模型"""
        from ultralytics import YOLO
        model = YOLO(self.model_path)
        if output_path is None:
            output_path = str(Path(self.model_path).with_suffix('.onnx'))
        logger.info(f"Exporting ONNX: {self.model_path} -> {output_path}")
        model.export(format='onnx', imgsz=imgsz, simplify=simplify, opset=opset)
        logger.info(f"ONNX exported: {output_path}")
        return output_path

    def export_tensorrt(self, output_path: str = None, imgsz: int = 640,
                         half: bool = True, int8: bool = False,
                         workspace: int = 4) -> str:
        """导出TensorRT引擎"""
        from ultralytics import YOLO
        model = YOLO(self.model_path)
        if output_path is None:
            suffix = '.engine'
            output_path = str(Path(self.model_path).with_suffix(suffix))
        logger.info(f"Exporting TensorRT: half={half}, int8={int8}")
        model.export(format='engine', imgsz=imgsz, half=half,
                      int8=int8, workspace=workspace)
        logger.info(f"TensorRT exported: {output_path}")
        return output_path

    def benchmark(self, imgsz: int = 640, iterations: int = 100) -> dict:
        """基准测试，测量推理延迟和吞吐"""
        from ultralytics import YOLO
        model = YOLO(self.model_path)
        dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        # 预热
        for _ in range(10):
            model.predict(dummy, device=self.device, verbose=False)
        # 测量
        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            model.predict(dummy, device=self.device, verbose=False)
            latencies.append((time.perf_counter() - t0) * 1000)
        result = {
            "model": self.model_path,
            "imgsz": imgsz,
            "iterations": iterations,
            "avg_ms": round(np.mean(latencies), 1),
            "p50_ms": round(np.percentile(latencies, 50), 1),
            "p95_ms": round(np.percentile(latencies, 95), 1),
            "p99_ms": round(np.percentile(latencies, 99), 1),
            "fps": round(1000.0 / np.mean(latencies), 1)
        }
        logger.info(f"Benchmark: avg={result['avg_ms']}ms, "
                     f"p95={result['p95_ms']}ms, fps={result['fps']}")
        return result


def get_optimal_weights(weights_dir: str = "weights/",
                         prefer_trt: bool = True) -> str:
    """自动选择最优权重格式：TensorRT > ONNX > PyTorch"""
    weights_dir = Path(weights_dir)
    if prefer_trt:
        for ext in ['*.engine', '*.trt']:
            found = list(weights_dir.glob(ext))
            if found:
                logger.info(f"Using TensorRT engine: {found[0]}")
                return str(found[0])
    for ext in ['*.onnx']:
        found = list(weights_dir.glob(ext))
        if found:
            logger.info(f"Using ONNX model: {found[0]}")
            return str(found[0])
    for ext in ['best.pt', '*.pt']:
        found = list(weights_dir.glob(ext))
        if found:
            logger.info(f"Using PyTorch model: {found[0]}")
            return str(found[0])
    raise FileNotFoundError(f"No model weights found in {weights_dir}")
