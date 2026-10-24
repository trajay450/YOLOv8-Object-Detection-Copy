"""Train Ultralytics YOLOv8 on the generated synthetic shapes."""
from __future__ import annotations
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/shape-yolo/dataset.yaml')
    parser.add_argument('--model', default='yolov8n.pt', help='Ultralytics YOLOv8 pretrained checkpoint')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--imgsz', type=int, default=256)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='cpu', help='cpu or device index e.g. 0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not Path(args.data).is_file():
        raise SystemExit('Dataset not found. Run python -m shape_detector.data first.')
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit('Install optional training dependency: pip install -e ".[train]"') from e
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, seed=args.seed,
                project='runs/shape-detector', name='train', exist_ok=False)

if __name__ == '__main__':
    main()
