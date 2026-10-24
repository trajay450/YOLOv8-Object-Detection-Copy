import cv2
import pytest
import yaml
from shape_detector.data import CLASSES, box_to_yolo, generate_sample, build_dataset


def test_bbox_conversion():
    assert box_to_yolo((0, 0, 50, 100), 100) == (0.25, 0.5, 0.5, 1.0)


def test_invalid_boxes():
    with pytest.raises(ValueError):
        box_to_yolo((4, 2, 4, 9), 100)
    with pytest.raises(ValueError):
        box_to_yolo((-1, 0, 2, 5), 100)


def test_reproducible_sample():
    a, labels_a, boxes_a = generate_sample(42)
    b, labels_b, boxes_b = generate_sample(42)
    assert (a == b).all()
    assert labels_a == labels_b and boxes_a == boxes_b


def test_sample_label_bounds():
    img, labels, boxes = generate_sample(12)
    assert img.shape == (256, 256, 3)
    assert 1 <= len(labels) <= 3
    for row in labels:
        parts = row.split()
        assert int(parts[0]) in range(len(CLASSES))
        assert len(parts) == 5
        assert all(0 < float(x) <= 1 for x in parts[1:])


def test_small_dataset(tmp_path):
    config = build_dataset(tmp_path / 'dataset', counts={'train': 3, 'val': 2, 'test': 1})
    cfg = yaml.safe_load(config.read_text())
    assert cfg['names'] == {0:'circle',1:'square',2:'triangle'}
    for split, n in [('train',3),('val',2),('test',1)]:
        images = sorted((config.parent / 'images' / split).glob('*.png'))
        labels = sorted((config.parent / 'labels' / split).glob('*.txt'))
        assert len(images) == len(labels) == n
        for im, txt in zip(images, labels):
            assert im.stem == txt.stem
            assert cv2.imread(str(im)).shape == (256,256,3)
            assert txt.read_text().strip()


def test_small_image_rejected():
    with pytest.raises(ValueError):
        generate_sample(42, size=100)
