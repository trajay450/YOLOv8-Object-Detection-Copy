"""Generate a deterministic, completely synthetic YOLO bounding-box dataset."""
from __future__ import annotations
import argparse
import random
from pathlib import Path
import cv2
import numpy as np
import yaml

CLASSES = ('circle', 'square', 'triangle')
SPLITS = {'train': 180, 'val': 45, 'test': 45}
COLORS = [(240, 120, 50), (50, 195, 215), (115, 105, 245)]  # BGR


def box_to_yolo(box: tuple[int, int, int, int], image_size: int) -> tuple[float, float, float, float]:
    """Convert pixel xyxy coordinates to YOLO normalized cxcywh."""
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= image_size and 0 <= y1 < y2 <= image_size):
        raise ValueError('Box must be nonempty and fully inside the image.')
    return ((x1 + x2) / (2 * image_size), (y1 + y2) / (2 * image_size),
            (x2 - x1) / image_size, (y2 - y1) / image_size)


def draw_shape(image: np.ndarray, class_id: int, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    color = COLORS[class_id]
    if class_id == 0:
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = ((x2 - x1) // 2, (y2 - y1) // 2)
        cv2.ellipse(image, center, axes, 0, 0, 360, color, -1, lineType=cv2.LINE_8)
    elif class_id == 1:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    else:
        pts = np.array([[(x1 + x2) // 2, y1], [x2, y2], [x1, y2]], np.int32)
        cv2.fillPoly(image, [pts], color)


def generate_sample(seed: int, size: int = 256) -> tuple[np.ndarray, list[str], list[tuple[int,tuple[int,int,int,int]]]]:
    """Create a synthetic scene with 1-3 nonoverlapping shapes and YOLO labels."""
    if size < 128:
        raise ValueError('Image size must be >= 128.')
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    image = np.full((size, size, 3), rng.randint(25, 45), dtype=np.uint8)
    noise = np_rng.integers(-8, 9, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    records: list[tuple[int,tuple[int,int,int,int]]] = []
    for _ in range(rng.randint(1, 3)):
        for _attempt in range(100):
            side = rng.randint(size // 7, size // 4)
            x1 = rng.randint(12, size - side - 13)
            y1 = rng.randint(12, size - side - 13)
            box = (x1, y1, x1 + side, y1 + side)
            # Prevent overlap and leave some separation for clean labels.
            if all(box[2]+8 < b[0] or b[2]+8 < box[0] or box[3]+8 < b[1] or b[3]+8 < box[1]
                   for _, b in records):
                cid = rng.randrange(len(CLASSES))
                draw_shape(image, cid, box)
                records.append((cid, box))
                break
    labels = []
    for cid, box in records:
        coords = box_to_yolo(box, size)
        labels.append(str(cid) + ' ' + ' '.join(f'{v:.6f}' for v in coords))
    return image, labels, records


def build_dataset(output: str | Path = 'data/shape-yolo', *, counts: dict[str, int] | None = None,
                  size: int = 256, seed: int = 42) -> Path:
    root = Path(output).resolve()
    chosen_counts = counts or SPLITS
    for split, count in chosen_counts.items():
        if split not in SPLITS or count < 0:
            raise ValueError('Expected nonnegative train/val/test counts.')
        (root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (root / 'labels' / split).mkdir(parents=True, exist_ok=True)
    for split_idx, (split, count) in enumerate(chosen_counts.items()):
        for i in range(count):
            img, labels, _ = generate_sample(seed + split_idx * 1_000_000 + i, size=size)
            filename = f'{split}_{i:04d}'
            ok = cv2.imwrite(str(root / 'images' / split / f'{filename}.png'), img)
            if not ok:
                raise IOError('Failed to write image.')
            (root / 'labels' / split / f'{filename}.txt').write_text('\n'.join(labels) + '\n', encoding='utf-8')
    config = {
        'path': str(root),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {i: v for i, v in enumerate(CLASSES)},
    }
    (root / 'dataset.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    return root / 'dataset.yaml'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/shape-yolo')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--train', type=int, default=180)
    parser.add_argument('--val', type=int, default=45)
    parser.add_argument('--test', type=int, default=45)
    args = parser.parse_args()
    cfg = build_dataset(args.output, counts={'train': args.train, 'val': args.val, 'test': args.test},
                        size=args.size, seed=args.seed)
    print(f'Dataset created: {cfg}')

if __name__ == '__main__':
    main()
