"""
模型导出脚本
用法:
  python scripts/export_model.py --weights weights/best.pt --format onnx
  python scripts/export_model.py --weights weights/best.pt --format engine --int8
  python scripts/export_model.py --weights weights/best.pt --benchmark
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from loguru import logger
from src.model_optimizer import ModelOptimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--format", choices=["onnx", "engine"], default="onnx")
    parser.add_argument("--half", action="store_true", help="FP16")
    parser.add_argument("--int8", action="store_true", help="INT8 quantization")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    optimizer = ModelOptimizer(args.weights)

    if args.benchmark:
        results = optimizer.benchmark(imgsz=args.imgsz)
        logger.info(f"Benchmark results: {results}")
        return

    if args.format == "onnx":
        path = optimizer.export_onnx(imgsz=args.imgsz)
    else:
        path = optimizer.export_tensorrt(
            imgsz=args.imgsz, half=args.half, int8=args.int8)
    logger.info(f"Exported: {path}")


if __name__ == "__main__":
    main()
