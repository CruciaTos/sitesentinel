"""Demo camera: posts made-up detections to the backend so the dashboard and metrics have live data.

Each frame has 0-4 people. A worn hard hat sits on the head and a worn vest on the torso. A missing
item sometimes still shows up nearby (hat held at the waist, vest on the ground, item next to the
person), so the backend's PPE model has to tell worn from merely visible.

Example:
    python simulator.py --count 10 --url http://localhost:8080

Settings: BACKEND_URL, CAMERA_IDS, INTERVAL_SECONDS, VIOLATION_RATE, SEED (environment variables).
Only the standard library is used, so it runs anywhere Python 3 does.
"""
import argparse
import http.client
import json
import os
import random
import signal
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://localhost:8080"
DEFAULT_CAMERAS = "gate-1,scaffold-2,crane-3"
# How likely 0, 1, 2, 3 or 4 people are in a frame.
PEOPLE_WEIGHTS = [1, 3, 3, 2, 1]
DISTRACTOR_RATE = 0.5
MAX_BACKOFF_SECONDS = 30


def _box(top, left, height, width):
    """A [ymin, xmin, ymax, xmax] box of the given size, moved inside the frame if it sticks out."""
    height, width = min(height, 1.0), min(width, 1.0)
    top = min(max(top, 0.0), 1.0 - height)
    left = min(max(left, 0.0), 1.0 - width)
    return [round(top, 4), round(left, 4), round(top + height, 4), round(left + width, 4)]


def _place(person, rel_cy, rel_cx, rel_h, rel_w):
    """An item box centred at (rel_cy, rel_cx), sized relative to the person's box."""
    top, left, bottom, right = person
    h, w = bottom - top, right - left
    item_h, item_w = rel_h * h, rel_w * w
    return _box(top + rel_cy * h - item_h / 2, left + rel_cx * w - item_w / 2, item_h, item_w)


def _worn_hardhat(rng, person):
    return _place(person, rng.uniform(0.03, 0.11), rng.uniform(0.45, 0.55),
                  rng.uniform(0.12, 0.18), rng.uniform(0.35, 0.5))


def _worn_vest(rng, person):
    return _place(person, rng.uniform(0.37, 0.45), rng.uniform(0.45, 0.55),
                  rng.uniform(0.28, 0.34), rng.uniform(0.75, 0.9))


def _beside(rng, person, rel_h, rel_w):
    """An item next to the person (on a barrier, a fence, the floor), outside their box."""
    gap = rng.uniform(0.02, 0.08)
    if rng.random() < 0.5:
        rel_cx = -gap - rel_w / 2
    else:
        rel_cx = 1 + gap + rel_w / 2
    return _place(person, rng.uniform(0.1, 0.9), rel_cx, rel_h, rel_w)


def _hardhat_distractor(rng, person):
    rel_h, rel_w = rng.uniform(0.12, 0.18), rng.uniform(0.35, 0.5)
    if rng.random() < 0.5:
        # Held in one hand at the waist.
        return _place(person, rng.uniform(0.45, 0.6), rng.choice([0.1, 0.9]), rel_h, rel_w)
    return _beside(rng, person, rel_h, rel_w)


def _vest_distractor(rng, person):
    if rng.random() < 0.5:
        # Lying flat on the ground at the person's feet.
        return _place(person, rng.uniform(0.95, 1.0), rng.uniform(0.45, 0.55),
                      rng.uniform(0.08, 0.12), rng.uniform(1.0, 1.2))
    return _beside(rng, person, rng.uniform(0.28, 0.34), rng.uniform(0.75, 0.9))


def _person(rng, slot_left, slot_width):
    """A person box inside one horizontal slot, leaving room on both sides for an item."""
    height = rng.uniform(0.35, 0.75)
    width = min(height * rng.uniform(0.25, 0.35), slot_width / 3)
    top = rng.uniform(0.0, 1.0 - height)
    left = slot_left + rng.uniform(width, slot_width - 2 * width)
    return _box(top, left, height, width)


def make_scene(rng, violation_rate, camera_id="gate-1", now_ms=None):
    """One made-up frame. Returns (payload for POST /v1/frames, what each person really wears)."""
    count = rng.choices(range(len(PEOPLE_WEIGHTS)), weights=PEOPLE_WEIGHTS)[0]
    persons, hardhats, vests, truth = [], [], [], []
    for i in range(count):
        # One person per slot, so people never overlap.
        person = _person(rng, i / count, 1.0 / count)
        wears = {"hardhat": True, "vest": True}
        if rng.random() < violation_rate:
            missing = rng.choice(["hardhat", "vest", "both"])
            wears["hardhat"] = missing == "vest"
            wears["vest"] = missing == "hardhat"

        if wears["hardhat"]:
            hardhats.append(_worn_hardhat(rng, person))
        elif rng.random() < DISTRACTOR_RATE:
            hardhats.append(_hardhat_distractor(rng, person))
        if wears["vest"]:
            vests.append(_worn_vest(rng, person))
        elif rng.random() < DISTRACTOR_RATE:
            vests.append(_vest_distractor(rng, person))
        persons.append(person)
        truth.append(wears)

    # The detector reports boxes in no particular order.
    rng.shuffle(hardhats)
    rng.shuffle(vests)
    payload = {
        "cameraId": camera_id,
        "timestamp": int(time.time() * 1000) if now_ms is None else now_ms,
        "persons": persons,
        "hardhats": hardhats,
        "vests": vests,
    }
    return payload, truth


def post(url, payload, timeout=5):
    """POST JSON to url. Returns (status code, decoded body); raises OSError if the server is unreachable."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as err:
        status, raw = err.code, err.read()
    try:
        body = json.loads(raw.decode("utf-8") or "null")
    except ValueError:
        body = None
    return status, body


def _describe(camera_id, truth, status, body):
    missing = sum(1 for t in truth if not (t["hardhat"] and t["vest"]))
    people = "1 person" if len(truth) == 1 else "{} people".format(len(truth))
    line = "[{}] {}, {} without full PPE -> {}".format(camera_id, people, missing, status)
    if status == 201 and isinstance(body, dict):
        flagged = body.get("people", 0) - body.get("protected", 0)
        line += " {} (model {} flagged {})".format(body.get("status"), body.get("model_version"), flagged)
    elif isinstance(body, dict) and body.get("message"):
        line += " " + body["message"]
    return line


def _log(line):
    print(line, flush=True)


def _deliver(post_fn, url, payload, sleep, log):
    """POST one frame, waiting and retrying for as long as the backend is unreachable."""
    backoff = 1
    while True:
        try:
            return post_fn(url, payload)
        except (OSError, http.client.HTTPException) as err:
            log("Backend not reachable at {} ({}); retrying in {}s".format(url, err, backoff))
            sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)


def run(base_url, cameras, interval, violation_rate, count=None, rng=None, post_fn=None, sleep=None, log=_log):
    """Send frames round-robin across cameras until `count` frames were delivered (forever if None)."""
    rng = rng or random.Random()
    post_fn = post_fn or post
    sleep = sleep or time.sleep
    url = base_url.rstrip("/") + "/v1/frames"
    sent = 0
    while count is None or sent < count:
        camera_id = cameras[sent % len(cameras)]
        payload, truth = make_scene(rng, violation_rate, camera_id)
        status, body = _deliver(post_fn, url, payload, sleep, log)
        sent += 1
        log(_describe(camera_id, truth, status, body))
        if count is None or sent < count:
            sleep(interval)
    return sent


def main(argv=None):
    parser = argparse.ArgumentParser(description="Send made-up PPE detections to the SiteSentinel backend.")
    parser.add_argument("--count", type=int, default=None,
                        help="send this many frames, then exit (default: run until stopped)")
    parser.add_argument("--url", default=os.getenv("BACKEND_URL", DEFAULT_URL),
                        help="backend base URL (default: $BACKEND_URL or {})".format(DEFAULT_URL))
    args = parser.parse_args(argv)

    cameras = [c.strip() for c in os.getenv("CAMERA_IDS", DEFAULT_CAMERAS).split(",") if c.strip()]
    interval = float(os.getenv("INTERVAL_SECONDS", 2))
    violation_rate = float(os.getenv("VIOLATION_RATE", 0.3))
    seed = os.getenv("SEED")
    rng = random.Random(int(seed)) if seed else random.Random()

    # Exit promptly on `docker stop` (as PID 1, Python would otherwise ignore SIGTERM).
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
    _log("Sending frames for {} to {} every {}s".format(", ".join(cameras), args.url, interval))
    try:
        run(args.url, cameras, interval, violation_rate, count=args.count, rng=rng)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
