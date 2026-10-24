"""Save visualized detections for an image or image folder."""
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--source', default='data/shape-yolo/images/test')
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()
    if not Path(args.weights).is_file():
        raise SystemExit('Missing trained weights file.')
    from ultralytics import YOLO
    YOLO(args.weights).predict(source=args.source, conf=args.conf, save=True,
                               project='runs/shape-detector', name='predict')

if __name__ == '__main__':
    main()
