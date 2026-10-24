"""Evaluate a trained YOLOv8 checkpoint on synthetic held-out test images."""
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True, help='Path to trained best.pt')
    parser.add_argument('--data', default='data/shape-yolo/dataset.yaml')
    parser.add_argument('--imgsz', type=int, default=256)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if not Path(args.weights).is_file():
        raise SystemExit('Checkpoint not found. Train the model and pass the actual best.pt path.')
    from ultralytics import YOLO
    metrics = YOLO(args.weights).val(data=args.data, split='test', imgsz=args.imgsz, device=args.device,
                                    project='runs/shape-detector', name='test')
    print(f'mAP@0.50: {metrics.box.map50:.4f}')
    print(f'mAP@0.50:0.95: {metrics.box.map:.4f}')
    print(f'Mean precision: {metrics.box.mp:.4f}')
    print(f'Mean recall: {metrics.box.mr:.4f}')

if __name__ == '__main__':
    main()
