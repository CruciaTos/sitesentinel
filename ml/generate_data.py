"""Generate synthetic labelled (person box, item box) pairs for the PPE association model.

Each row pairs one person box with one hard hat or vest box; label 1 means the person wears it.
Positives sit on the head or torso. Negatives cover what the old overlap rule gets wrong (an item
held at the waist, lying by the feet, worn by an overlapping neighbour) and items far away. Some
people are cropped by the frame edge, some items are unusually small or large, and about 3% of
labels are flipped, like real annotation mistakes.

Boxes are normalized [ymin, xmin, ymax, xmax]. The same seed always gives the same file.
"""
import argparse
import csv
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join("ml", "data", "ppe_pairs.csv")
COLUMNS = ["item_type", "p_ymin", "p_xmin", "p_ymax", "p_xmax",
           "i_ymin", "i_xmin", "i_ymax", "i_xmax", "label", "scenario"]
ITEM_TYPES = ("hardhat", "vest")
# Where a worn item's centre sits, as a fraction of the person's height from the top.
WORN_Y = {"hardhat": 0.08, "vest": 0.40}
NEGATIVES = (("held", 0.25), ("ground", 0.15), ("neighbour", 0.3), ("far", 0.3))
LABEL_NOISE = 0.03
CROP_RATE = 0.12
RESIZE_RATE = 0.1
# A detector only reports an item if enough of it is inside the frame.
MIN_VISIBLE = 0.4


def _box(cy, cx, h, w):
    return [cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2]


def _clip(box):
    return [min(max(v, 0.0), 1.0) for v in box]


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _pick(rng, weighted):
    r = rng.random()
    for name, weight in weighted:
        r -= weight
        if r < 0:
            return name
    return weighted[-1][0]


def _person(rng, cropped):
    """(top, left, height, width) of a person; a cropped one runs off the frame edge."""
    h = rng.uniform(0.2, 0.85)
    w = h * rng.uniform(0.28, 0.5)
    top = rng.uniform(0.0, 1.0 - h)
    left = rng.uniform(0.0, 1.0 - w)
    if cropped:
        side = rng.choice(("bottom", "left", "right"))
        if side == "bottom":
            top = 1.0 - h * rng.uniform(0.6, 0.9)
        elif side == "left":
            left = -w * rng.uniform(0.1, 0.4)
        else:
            left = 1.0 - w * rng.uniform(0.6, 0.9)
    return top, left, h, w


def _item_size(rng, item_type, scenario, h, w):
    if item_type == "hardhat":
        return h * rng.uniform(0.07, 0.13), w * rng.uniform(0.25, 0.5)
    if scenario == "held":  # folded up in a hand
        return h * rng.uniform(0.12, 0.25), w * rng.uniform(0.3, 0.6)
    if scenario == "ground":  # lying flat
        return h * rng.uniform(0.05, 0.12), w * rng.uniform(0.6, 1.2)
    return h * rng.uniform(0.22, 0.38), w * rng.uniform(0.6, 1.0)


def _worn_center(rng, item_type, top, left, h, w):
    spread = 0.025 if item_type == "hardhat" else 0.04
    cy = top + h * (WORN_Y[item_type] + rng.gauss(0.0, spread))
    cx = left + w * (0.5 + rng.gauss(0.0, 0.07))
    return cy, cx


def _sample(rng, item_type, label):
    """One row, or None if the item ended up (mostly) outside the frame."""
    scenario = "worn" if label else _pick(rng, NEGATIVES)
    cropped = rng.random() < CROP_RATE
    top, left, h, w = _person(rng, cropped)
    ih, iw = _item_size(rng, item_type, scenario, h, w)
    size = ""
    if rng.random() < RESIZE_RATE:
        size = rng.choice(("small", "large"))
        factor = rng.uniform(0.5, 0.7) if size == "small" else rng.uniform(1.4, 1.9)
        ih, iw = ih * factor, iw * factor

    if scenario == "worn":
        cy, cx = _worn_center(rng, item_type, top, left, h, w)
    elif scenario == "held":
        low, high = (0.45, 0.7) if item_type == "hardhat" else (0.55, 0.8)
        cy = top + h * rng.uniform(low, high)
        hand = rng.random()
        if hand < 0.35:
            cx = left + w * rng.uniform(-0.05, 0.25)
        elif hand < 0.7:
            cx = left + w * rng.uniform(0.75, 1.05)
        else:
            cx = left + w * rng.uniform(0.35, 0.65)
    elif scenario == "ground":
        cy = top + h * rng.uniform(0.9, 1.04)
        cx = left + w * rng.uniform(-0.4, 1.4)
    elif scenario == "neighbour":
        scale = rng.uniform(0.8, 1.2)
        nh, nw = h * scale, w * scale
        if rng.random() < 0.3:  # up on a scaffold above this person
            n_top = top - nh * rng.uniform(0.7, 1.4)
            n_left = left + (w - nw) / 2 + w * rng.uniform(-0.6, 0.6)
        else:  # standing next to this person
            n_top = top + h - nh + h * rng.uniform(-0.08, 0.08)
            n_left = left + (w - nw) / 2 + w * rng.uniform(0.35, 1.1) * rng.choice((-1, 1))
        ih, iw = ih * scale, iw * scale
        cy, cx = _worn_center(rng, item_type, n_top, n_left, nh, nw)
    else:  # far: anywhere else in the frame, e.g. on someone across the site or up on a scaffold
        cy, cx = rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)
        if abs(cx - (left + w / 2)) < w and abs(cy - (top + h / 2)) < h:
            return None

    item = _box(cy, cx, ih, iw)
    visible = _clip(item)
    if _area(visible) < MIN_VISIBLE * _area(item):
        return None
    person = _clip([top, left, top + h, left + w])
    if rng.random() < LABEL_NOISE:
        label = 1 - label
    scenario += "+cropped" if cropped else ""
    scenario += "+" + size if size else ""
    return [item_type] + person + visible + [label, scenario]


def generate(rows=6000, seed=42):
    """Rows as lists in COLUMNS order, balanced per item type and label (before label noise)."""
    rng = random.Random(seed)
    out = []
    for i in range(rows):
        item_type = ITEM_TYPES[i % 2]
        label = 1 if (i // 2) % 2 == 0 else 0
        row = None
        while row is None:
            row = _sample(rng, item_type, label)
        out.append(row)
    return out


def write_csv(rows, path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([row[0]] + ["{:.4f}".format(v) for v in row[1:9]] + row[9:])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=6000, help="number of pairs (default 6000)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="CSV to write; relative paths are taken from the repo root")
    args = parser.parse_args(argv)
    if args.rows < 1:
        parser.error("--rows must be at least 1")
    path = os.path.join(ROOT, args.out)
    rows = generate(args.rows, args.seed)
    write_csv(rows, path)
    positives = sum(row[9] for row in rows)
    print("[MLOps] Wrote {} pairs ({} worn, {} not worn, seed {}) to {}".format(
        len(rows), positives, len(rows) - positives, args.seed, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
