"""Decide whether each detected person is wearing a hard hat and a vest.

Boxes are normalized [ymin, xmin, ymax, xmax], as the detection model returns them.
"""
import numpy as np

HARDHAT, VEST, PERSON = 1, 2, 3
MIN_OVERLAP = 0.6


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _share_inside(item_box, person_box):
    """Fraction of the item's box that lies inside the person's box."""
    top = max(item_box[0], person_box[0])
    left = max(item_box[1], person_box[1])
    bottom = min(item_box[2], person_box[2])
    right = min(item_box[3], person_box[3])
    item_area = _area(item_box)
    if item_area <= 0:
        return 0.0
    return max(0.0, bottom - top) * max(0.0, right - left) / item_area


def wears(person_box, item_boxes, min_overlap=MIN_OVERLAP):
    return any(_share_inside(item, person_box) > min_overlap for item in item_boxes)


def split_detections(output_dict, min_score):
    """Return (hardhat_boxes, vest_boxes, person_boxes) above the score threshold."""
    keep = output_dict["detection_scores"] > min_score
    boxes = output_dict["detection_boxes"][keep]
    classes = output_dict["detection_classes"][keep]
    return (
        boxes[np.where(classes == HARDHAT)],
        boxes[np.where(classes == VEST)],
        boxes[np.where(classes == PERSON)],
    )


def assess_people(output_dict, min_score):
    """One entry per person: {"box", "hardhat", "vest"}."""
    hardhats, vests, people = split_detections(output_dict, min_score)
    return [
        {"box": box, "hardhat": wears(box, hardhats), "vest": wears(box, vests)}
        for box in people
    ]
