"""The JSON messages the detector posts to the backend."""
import time

from compliance import split_detections


def _now_ms():
    return int(time.time() * 1000)


def _boxes(array):
    return [[round(float(v), 4) for v in box] for box in array]


def frame_message(camera_id, output_dict, min_score, image=None, now_ms=None):
    """Boxes for POST /v1/frames. The backend's trained model decides who wears what."""
    hardhats, vests, persons = split_detections(output_dict, min_score)
    message = {
        "cameraId": camera_id,
        "timestamp": now_ms if now_ms is not None else _now_ms(),
        "persons": _boxes(persons),
        "hardhats": _boxes(hardhats),
        "vests": _boxes(vests),
    }
    if image:
        message["image"] = image
    return message


def detection_message(camera_id, people, image=None, now_ms=None):
    """Per-person decisions for POST /v1/detections (the original contract; the overlap rule decides here)."""
    message = {
        "cameraId": camera_id,
        "timestamp": now_ms if now_ms is not None else _now_ms(),
        "persons": [{"hardhat": bool(p["hardhat"]), "vest": bool(p["vest"])} for p in people],
    }
    if image:
        message["image"] = image
    return message
