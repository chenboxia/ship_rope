"""
训练脚本
用法:
  # 4类统一模型（系缆+救生衣，推荐）
  python scripts/train.py

  # 从预训练权重微调
  python scripts/train.py --pretrained weights/pretrained.pt --epochs 50
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from loguru import logger
from src.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train YOLO11s mooring model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--data", type=str, default="data/dataset.yaml")
    parser.add_argument("--model", type=str, default="yolo11s")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="mooring_yolo11s")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("Mooring Detection - Training")
    logger.info("=" * 50)

    trainer = Trainer({"model_name": args.model, "data_yaml": args.data,
                        "project": args.project, "name": args.name})

    if args.validate_only:
        if not args.pretrained:
            logger.error("--pretrained required for validation")
            return
        trainer.validate(args.pretrained, device=args.device)
        return

    best = trainer.train(
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        lr0=args.lr0, patience=args.patience, device=args.device,
        workers=args.workers, resume=bool(args.resume),
        pretrained_weights=args.pretrained or args.resume)
    logger.info("Training complete. Best: %s" % best)
    if os.path.exists(best):
        logger.info("Running validation...")
        trainer.validate(best, device=args.device)

if __name__ == "__main__":
    main()
