# YOLOv8 Object Detection — Synthetic Shapes Benchmark

![Sample synthetic detection dataset](assets/synthetic_examples.png)

A compact, reproducible **computer vision engineering project** by **Ajay Kumar Thakur**. Demonstrates object-detection dataset construction, bounding-box conversion, YOLOv8 training, held-out evaluation, and inference visualization. This is a **standalone educational project**, separate from unpublished research manuscripts.

> **Evaluation status:** Dataset generation and annotation tests run locally. **YOLOv8 training has NOT been run or evaluated in this release**; no detector accuracy or mAP results are claimed. Training downloads pretrained weights on first run and requires the optional `ultralytics` package.

## Features

- Procedurally generated 256×256 RGB scenes, containing **circles, squares, and triangles**.
- YOLO normalized `class cx cy width height` annotation files.
- Nonoverlapping objects; reproducible generation from a seed.
- Disjoint images across fixed train/validation/test sets (180 / 45 / 45 by default).
- Ultralytics YOLOv8 nano model (`yolov8n.pt`) as an optional training backbone.
- Real mAP@0.50, mAP@0.50:0.95, precision and recall output **after you train and evaluate**.
- Dataset unit tests and GitHub Actions CI.

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
# On macOS/Linux:
source .venv/bin/activate
# On Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -e ".[train,test]"
python -m pytest -q
python -m shape_detector.data
python -m shape_detector.train --epochs 15 --device cpu
```

**Training output:** Ultralytics prints and creates a run directory such as `runs/shape-detector/train/weights/best.pt`. Depending on existing runs, the directory name may differ. Find the actual `best.pt`, then run:

```bash
python -m shape_detector.evaluate --weights runs/shape-detector/train/weights/best.pt
python -m shape_detector.predict --weights runs/shape-detector/train/weights/best.pt
```

Training can be faster on a CUDA GPU: `--device 0`. CPU training may take considerably longer. **The first YOLO run may require internet access to download pretrained weights.** If downloading weights is unavailable, use a local checkpoint through `--model /path/to/weights.pt`.

### Quick smoke test without training

```bash
pip install -e ".[test]"
python -m pytest -q
python -m shape_detector.data --train 6 --val 3 --test 3 --output data/quick-demo
```

## Folder structure

```text
src/shape_detector/
  data.py          # Generate images, YOLO labels, YAML config
  train.py         # YOLOv8n training CLI
  evaluate.py      # Held-out test performance (requires trained weights)
  predict.py       # Save visualized predictions (requires trained weights)
tests/             # Synthetic data and bounding-box tests
assets/            # Example synthetic source images
.github/workflows/ # Automated tests
```

Generated images, model weights, and training runs are excluded from Git using `.gitignore`.

## Protocol and limitations

1. Generate data using a fixed seed (`42` by default). Each split has a separate seed range.
2. Train on `train`; monitor `val`; evaluate once on `test` using trained weights.
3. Report mAP@.50, mAP@.50:.95, precision, and recall, along with the seed and software versions.
4. Synthetic geometric objects are deliberately simple. Strong benchmark results would **not** establish real-world object-detection reliability or generalization.
5. The images are generated in code and contain no patient, medical, personal, or licensed third-party imagery.
6. The shapes and annotations are pedagogical; output masks or bounding boxes may contain pixel-edge approximations.

## Responsible reuse

Code in this repository is MIT licensed. The Ultralytics package and pretrained weights have **separate licensing terms**; review their current license before redistributing weights or incorporating them into commercial software.

## Suggested extensions

- Compare random initialization with pretrained weights.
- Vary background noise and quantify mAP changes.
- Test performance under occlusion and domain shifts.
- Build a Streamlit inference demo for uploaded non-sensitive images.

## Author

**Ajay Kumar Thakur** · PhD researcher, AI & Big Data Convergence, Seoul Christian University  
GitHub: [@trajay450](https://github.com/trajay450)
