"""Draws the live video overlay: corner-bracket boxes, status chips and a summary strip."""
import time

import cv2

from compliance import split_detections

# OpenCV colours are BGR.
SAFE = (74, 160, 23)        # green
UNSAFE = (46, 16, 200)      # red
TAPE = (0, 196, 255)        # safety yellow
INK = (31, 26, 21)
PAPER = (241, 242, 238)
FONT = cv2.FONT_HERSHEY_DUPLEX


def _to_pixels(box, width, height):
    ymin, xmin, ymax, xmax = box
    return int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)


def _brackets(img, p1, p2, color, thickness, arm):
    (x1, y1), (x2, y2) = p1, p2
    arm = max(8, min(arm, (x2 - x1) // 2, (y2 - y1) // 2))
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * arm, cy), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * arm), color, thickness, cv2.LINE_AA)


def _chip(img, text, origin, bg, fg, scale):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    x, y = origin
    pad = 5
    top = max(0, y - th - base - pad * 2)
    cv2.rectangle(img, (x, top), (x + tw + pad * 2, top + th + base + pad * 2), bg, -1)
    cv2.putText(img, text, (x + pad, top + th + pad), FONT, scale, fg, 1, cv2.LINE_AA)


def _label(person):
    if person["hardhat"] and person["vest"]:
        return "Protected"
    if not person["hardhat"] and not person["vest"]:
        return "No hard hat, no vest"
    return "No hard hat" if not person["hardhat"] else "No vest"


def draw_frame(frame, output_dict, people, camera_id, min_score):
    """Draw detections onto `frame` in place. `people` comes from compliance.assess_people."""
    height, width = frame.shape[:2]
    scale = max(0.45, width / 1600.0)
    thick = max(2, width // 400)

    hardhats, vests, _ = split_detections(output_dict, min_score)
    for box in hardhats:
        x1, y1, x2, y2 = _to_pixels(box, width, height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), TAPE, 1, cv2.LINE_AA)
    for box in vests:
        x1, y1, x2, y2 = _to_pixels(box, width, height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), TAPE, 1, cv2.LINE_AA)

    for person in people:
        x1, y1, x2, y2 = _to_pixels(person["box"], width, height)
        ok = person["hardhat"] and person["vest"]
        color = SAFE if ok else UNSAFE
        _brackets(frame, (x1, y1), (x2, y2), color, thick + 1, arm=(x2 - x1) // 4)
        _chip(frame, _label(person), (x1, y1), color, PAPER, scale)

    _summary_strip(frame, people, camera_id, scale)
    return frame


def _summary_strip(frame, people, camera_id, scale):
    height, width = frame.shape[:2]
    bar_h = int(46 * max(scale * 1.6, 0.9))
    shade = frame.copy()
    cv2.rectangle(shade, (0, height - bar_h), (width, height), INK, -1)
    cv2.addWeighted(shade, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (0, height - bar_h), (int(width * 0.012), height), TAPE, -1)

    protected = sum(1 for p in people if p["hardhat"] and p["vest"])
    at_risk = len(people) - protected
    text_y = height - bar_h // 2 + int(8 * scale * 1.6)
    x = int(width * 0.03)
    cv2.putText(frame, camera_id, (x, text_y), FONT, scale * 1.1, PAPER, 1, cv2.LINE_AA)

    if not people:
        status, color = "Nobody in view", PAPER
    elif at_risk:
        status, color = "{} of {} need attention".format(at_risk, len(people)), (120, 130, 255)
    else:
        status, color = "All {} protected".format(len(people)), (130, 220, 150)
    (tw, _), _ = cv2.getTextSize(status, FONT, scale * 1.1, 1)
    cv2.putText(frame, status, (width // 2 - tw // 2, text_y), FONT, scale * 1.1, color, 1, cv2.LINE_AA)

    stamp = time.strftime("%H:%M:%S")
    (sw, _), _ = cv2.getTextSize(stamp, FONT, scale * 1.1, 1)
    cv2.putText(frame, stamp, (width - sw - int(width * 0.03), text_y), FONT, scale * 1.1, PAPER, 1, cv2.LINE_AA)
