import numpy as np

from compliance import HARDHAT, PERSON, VEST
from messages import detection_message, frame_message


def detections(rows):
    """Build a model output dict from (class, score, box) rows."""
    return {
        "detection_classes": np.array([r[0] for r in rows], dtype=np.int64),
        "detection_scores": np.array([r[1] for r in rows], dtype=np.float32),
        "detection_boxes": np.array([r[2] for r in rows], dtype=np.float32),
    }


OUTPUT = detections([
    (PERSON, 0.95, [0.2, 0.3, 0.9, 0.5]),
    (HARDHAT, 0.90, [0.18, 0.36, 0.27, 0.44]),
    (VEST, 0.88, [0.35, 0.31, 0.6, 0.49]),
    (VEST, 0.20, [0.5, 0.6, 0.7, 0.8]),        # below the threshold: dropped
])


def test_frame_message_sends_boxes_above_the_threshold():
    msg = frame_message("gate-1", OUTPUT, 0.5, now_ms=1760000000000)
    assert msg["cameraId"] == "gate-1"
    assert msg["timestamp"] == 1760000000000
    assert msg["persons"] == [[0.2, 0.3, 0.9, 0.5]]
    assert msg["hardhats"] == [[0.18, 0.36, 0.27, 0.44]]
    assert msg["vests"] == [[0.35, 0.31, 0.6, 0.49]]
    assert "image" not in msg


def test_frame_message_boxes_are_plain_json_numbers():
    msg = frame_message("gate-1", OUTPUT, 0.5)
    for key in ("persons", "hardhats", "vests"):
        for box in msg[key]:
            assert len(box) == 4 and all(type(v) is float for v in box)
    assert isinstance(msg["timestamp"], int)


def test_frame_message_carries_the_snapshot():
    image = {"height": 360, "width": 640, "format": "jpeg", "raw": "abc="}
    assert frame_message("gate-1", OUTPUT, 0.5, image=image)["image"] == image


def test_detection_message_keeps_the_original_contract():
    people = [{"box": None, "hardhat": np.bool_(True), "vest": np.bool_(False)}]
    msg = detection_message("gate-1", people, now_ms=5)
    assert msg == {"cameraId": "gate-1", "timestamp": 5, "persons": [{"hardhat": True, "vest": False}]}
    assert type(msg["persons"][0]["hardhat"]) is bool
