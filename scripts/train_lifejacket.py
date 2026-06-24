"""
救生衣检测模型训练脚本
用法:
  python scripts/train_lifejacket.py
  python scripts/train_lifejacket.py --epochs 80 --batch 16
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from loguru import logger
from src.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/lifejacket_dataset.yaml")
    parser.add_argument("--model", type=str, default="yolo11s")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--pretrained", type=str, default="")
    args = parser.parse_args()
    logger.info("Lifejacket Detection - Training")
    trainer = Trainer({"model_name": args.model, "data_yaml": args.data,
                        "project": "runs/train", "name": "lifejacket_yolo11s"})
    best = trainer.train(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                          device=args.device, pretrained_weights=args.pretrained)
    logger.info(f"Best weights: {best}")
    if os.path.exists(best):
        trainer.validate(best, device=args.device)

if __name__ == "__main__":
    main()
