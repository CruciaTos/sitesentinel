"""In-memory record of recent safety checks, used by the dashboard."""
import base64
import binascii
import threading
import time
import uuid
from collections import deque

HOUR_MS = 3600 * 1000


class EventStore:
    def __init__(self, max_events=200, window_minutes=15, live_seconds=15):
        self._events = deque(maxlen=max_events)
        self._images = {}
        self._lock = threading.Lock()
        self.window_ms = window_minutes * 60 * 1000
        self.live_ms = live_seconds * 1000

    def add(self, camera_id, timestamp_ms, persons, image_b64=None):
        """Record one check. `persons` is a list of {"hardhat": bool, "vest": bool}."""
        both = sum(1 for p in persons if not p["hardhat"] and not p["vest"])
        no_hat = sum(1 for p in persons if not p["hardhat"] and p["vest"])
        no_vest = sum(1 for p in persons if p["hardhat"] and not p["vest"])
        total = len(persons)
        event = {
            "id": uuid.uuid4().hex[:12],
            "camera": camera_id,
            "ts": int(timestamp_ms),
            "people": total,
            "protected": total - both - no_hat - no_vest,
            "no_hardhat": no_hat,
            "no_vest": no_vest,
            "no_both": both,
            "status": "violation" if (both or no_hat or no_vest) else "clear",
            "has_image": False,
        }
        image = None
        if image_b64:
            try:
                image = base64.b64decode(image_b64)
                event["has_image"] = True
            except (binascii.Error, ValueError):
                image = None
        with self._lock:
            if len(self._events) == self._events.maxlen:
                dropped = self._events[0]
                self._images.pop(dropped["id"], None)
            self._events.append(event)
            if image is not None:
                self._images[event["id"]] = image
        return event

    def image(self, event_id):
        with self._lock:
            return self._images.get(event_id)

    def events(self, limit=50, violations_only=False):
        with self._lock:
            items = list(self._events)
        if violations_only:
            items = [e for e in items if e["status"] == "violation"]
        items.sort(key=lambda e: e["ts"], reverse=True)
        return items[:limit]

    def summary(self, now_ms=None):
        now_ms = int(time.time() * 1000) if now_ms is None else now_ms
        with self._lock:
            items = list(self._events)

        recent = [e for e in items if now_ms - e["ts"] <= self.window_ms]
        people = sum(e["people"] for e in recent)
        protected = sum(e["protected"] for e in recent)
        rate = round(100.0 * protected / people) if people else None

        cameras = {}
        for e in items:
            cam = cameras.setdefault(e["camera"], {"id": e["camera"], "last_seen": 0, "checks": 0})
            cam["last_seen"] = max(cam["last_seen"], e["ts"])
            cam["checks"] += 1
        for cam in cameras.values():
            cam["live"] = now_ms - cam["last_seen"] <= self.live_ms

        current_hour = now_ms // HOUR_MS
        hourly = []
        for h in range(current_hour - 11, current_hour + 1):
            bucket = [e for e in items if e["ts"] // HOUR_MS == h]
            hourly.append({
                "hour_start": h * HOUR_MS,
                "checks": len(bucket),
                "violations": sum(1 for e in bucket if e["status"] == "violation"),
            })

        return {
            "window_minutes": self.window_ms // 60000,
            "people": people,
            "protected": protected,
            "protected_rate": rate,
            "no_hardhat": sum(e["no_hardhat"] for e in recent),
            "no_vest": sum(e["no_vest"] for e in recent),
            "no_both": sum(e["no_both"] for e in recent),
            "checks_total": len(items),
            "cameras": sorted(cameras.values(), key=lambda c: c["id"]),
            "hourly": hourly,
            "server_time": now_ms,
        }
