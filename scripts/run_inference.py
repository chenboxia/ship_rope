"""
推理入口脚本
用法:
  python scripts/run_inference.py --source 0
  python scripts/run_inference.py --source video.mp4 --show
  python scripts/run_inference.py --source cam0.mp4 --source cam1.mp4
  python scripts/run_inference.py --source rtsp://192.168.1.10/stream --show
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from loguru import logger
from src.engine import Engine


def main():
    parser = argparse.ArgumentParser(description="Mooring monitoring inference")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--source", type=str, nargs="+", required=True,
                        help="Video source(s): file path, camera index (0), RTSP URL")
    parser.add_argument("--show", action="store_true", help="Show visualization")
    parser.add_argument("--weights", type=str, default="")
    args = parser.parse_args()

    sources = []
    for s in args.source:
        try:
            sources.append(int(s))
        except ValueError:
            sources.append(s)

    engine = Engine(config_path=args.config)
    if args.weights:
        engine._config_watcher.config["model"]["weights"] = args.weights
    engine.run(sources=sources, show=args.show)


if __name__ == "__main__":
    main()
