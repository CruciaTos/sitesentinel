"""PPE association model: decides whether a hard hat or vest box belongs to a person box.

The detector finds people, hard hats and vests as separate boxes. This module scores each
(person, item) pair. ml/train.py fits the model and writes model/ppe_model.json; the same
features() function is used for training and serving, so the two cannot drift apart.

Boxes are normalized [ymin, xmin, ymax, xmax], as the detection model returns them.
"""
import json
import math
import os

ITEM_TYPES = ("hardhat", "vest")
# Where each item usually sits on a person, as a fraction of the person's height from the top.
EXPECTED_CENTER_Y = {"hardhat": 0.08, "vest": 0.40}
FEATURES = ["share_inside", "iou", "rel_cy", "cy_dist", "cx_dist", "rel_w", "rel_h"]
# Item positions are clipped to one person-size around the box and sizes to 3x the person, so every
# feature is bounded: "far away" is just far away, and a linear model cannot extrapolate past it.
POSITION_RANGE = (-1.0, 2.0)
MAX_REL_SIZE = 3.0
# The original hand-written rule: an item is worn if most of its box lies inside the person's.
RULE_MIN_OVERLAP = 0.6
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "ppe_model.json")


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a, b):
    top, left = max(a[0], b[0]), max(a[1], b[1])
    bottom, right = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, bottom - top) * max(0.0, right - left)


def features(person_box, item_box, item_type):
    """Feature vector for one (person, item) pair, in the order of FEATURES."""
    if item_type not in EXPECTED_CENTER_Y:
        raise ValueError("item_type must be one of {}".format(", ".join(ITEM_TYPES)))
    person_h = max(person_box[2] - person_box[0], 1e-6)
    person_w = max(person_box[3] - person_box[1], 1e-6)
    inter = _intersection(person_box, item_box)
    item_area = _area(item_box)
    union = _area(person_box) + item_area - inter
    low, high = POSITION_RANGE
    rel_cy = min(max(((item_box[0] + item_box[2]) / 2.0 - person_box[0]) / person_h, low), high)
    rel_cx = min(max(((item_box[1] + item_box[3]) / 2.0 - person_box[1]) / person_w, low), high)
    return [
        inter / item_area if item_area > 0 else 0.0,
        inter / union if union > 0 else 0.0,
        rel_cy,
        abs(rel_cy - EXPECTED_CENTER_Y[item_type]),
        abs(rel_cx - 0.5),
        min((item_box[3] - item_box[1]) / person_w, MAX_REL_SIZE),
        min((item_box[2] - item_box[0]) / person_h, MAX_REL_SIZE),
    ]


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class _Model:
    name = "ppe-association"
    version = None
    algorithm = None
    trained_at = None
    threshold = 0.5
    metrics = {}

    def probability(self, person_box, item_box, item_type):
        raise NotImplementedError

    def wears(self, person_box, item_box, item_type):
        return self.probability(person_box, item_box, item_type) >= self.threshold

    def info(self):
        return {
            "name": self.name,
            "version": self.version,
            "algorithm": self.algorithm,
            "trained_at": self.trained_at,
            "metrics": self.metrics,
        }


class RuleModel(_Model):
    """Fallback used when no trained model is available: the original overlap rule."""
    version = "rules-0.6"
    algorithm = "overlap_rule"

    def probability(self, person_box, item_box, item_type):
        share_inside = features(person_box, item_box, item_type)[0]
        return 1.0 if share_inside > RULE_MIN_OVERLAP else 0.0


class LogisticModel(_Model):
    """One standardized logistic regression per item type, as exported by ml/train.py."""
    algorithm = "logistic_regression"

    def __init__(self, spec):
        if not isinstance(spec, dict):
            raise ValueError("model file must hold a JSON object")
        if spec.get("features") != FEATURES:
            raise ValueError("model was trained on different features")
        self.name = spec.get("name", self.name)
        self.version = spec["version"]
        self.algorithm = spec.get("algorithm", self.algorithm)
        self.trained_at = spec.get("trained_at")
        self.threshold = float(spec.get("threshold", 0.5))
        self.metrics = spec.get("metrics", {})
        self.models = {}
        for item in ITEM_TYPES:
            m = spec["models"][item]
            params = (m["mean"], m["scale"], m["coef"])
            if any(len(p) != len(FEATURES) for p in params):
                raise ValueError("model for {} has the wrong number of weights".format(item))
            self.models[item] = {
                "mean": [float(v) for v in m["mean"]],
                "scale": [float(v) or 1.0 for v in m["scale"]],
                "coef": [float(v) for v in m["coef"]],
                "intercept": float(m["intercept"]),
            }

    def probability(self, person_box, item_box, item_type):
        m = self.models[item_type]
        x = features(person_box, item_box, item_type)
        z = m["intercept"] + sum(
            c * (v - mu) / s for c, v, mu, s in zip(m["coef"], x, m["mean"], m["scale"])
        )
        return _sigmoid(z)


def load(path=None):
    """Load the trained model, or fall back to the overlap rule if it is missing or invalid."""
    path = path or DEFAULT_PATH
    try:
        with open(path, encoding="utf-8") as f:
            return LogisticModel(json.load(f))
    except (OSError, ValueError, KeyError, TypeError) as err:
        print("[Warn] PPE model not loaded from {} ({}); using the overlap rule".format(path, err))
        return RuleModel()


def assess_people(model, persons, hardhats, vests):
    """One {"hardhat": bool, "vest": bool} entry per person box."""
    return [
        {
            "hardhat": any(model.wears(p, h, "hardhat") for h in hardhats),
            "vest": any(model.wears(p, v, "vest") for v in vests),
        }
        for p in persons
    ]
