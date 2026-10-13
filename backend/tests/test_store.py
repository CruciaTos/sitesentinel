import base64

from store import HOUR_MS, EventStore

MINUTE_MS = 60 * 1000


def person(hardhat, vest):
    return {"hardhat": hardhat, "vest": vest}


def test_add_counts_each_kind_of_violation():
    store = EventStore()
    event = store.add("gate-1", 1000, [person(True, True), person(False, True), person(True, False),
                                       person(False, False)])
    assert event["camera"] == "gate-1"
    assert event["ts"] == 1000
    assert event["people"] == 4
    assert event["protected"] == 1
    assert (event["no_hardhat"], event["no_vest"], event["no_both"]) == (1, 1, 1)
    assert event["status"] == "violation"
    assert event["has_image"] is False


def test_everyone_protected_or_nobody_seen_is_clear():
    store = EventStore()
    assert store.add("gate-1", 1, [person(True, True)] * 3)["status"] == "clear"
    empty = store.add("gate-1", 2, [])
    assert empty["status"] == "clear"
    assert empty["people"] == 0


def test_eviction_drops_oldest_event_and_its_image():
    store = EventStore(max_events=2)
    image = base64.b64encode(b"jpeg-bytes").decode()
    first = store.add("gate-1", 1, [], image)
    second = store.add("gate-1", 2, [], image)
    third = store.add("gate-1", 3, [], image)
    assert store.image(first["id"]) is None
    assert store.image(second["id"]) == b"jpeg-bytes"
    assert store.image(third["id"]) == b"jpeg-bytes"
    assert [e["id"] for e in store.events()] == [third["id"], second["id"]]


def test_bad_base64_image_is_ignored():
    store = EventStore()
    event = store.add("gate-1", 1, [person(True, True)], "not base64!")
    assert event["has_image"] is False
    assert store.image(event["id"]) is None


def test_events_newest_first_with_filter_and_limit():
    store = EventStore()
    store.add("a", 100, [person(True, True)])
    bad = store.add("a", 300, [person(False, False)])
    store.add("a", 200, [person(True, True)])
    assert [e["ts"] for e in store.events()] == [300, 200, 100]
    assert [e["id"] for e in store.events(violations_only=True)] == [bad["id"]]
    assert [e["ts"] for e in store.events(limit=2)] == [300, 200]


def test_summary_rate_counts_only_the_window():
    store = EventStore(window_minutes=15)
    now = 50 * HOUR_MS
    store.add("a", now - MINUTE_MS, [person(True, True), person(True, True), person(False, True)])
    store.add("a", now - 20 * MINUTE_MS, [person(False, False)])
    summary = store.summary(now_ms=now)
    assert summary["window_minutes"] == 15
    assert summary["people"] == 3
    assert summary["protected"] == 2
    assert summary["protected_rate"] == 67
    assert (summary["no_hardhat"], summary["no_vest"], summary["no_both"]) == (1, 0, 0)
    assert summary["checks_total"] == 2
    assert summary["server_time"] == now


def test_summary_without_people_has_no_rate():
    store = EventStore()
    store.add("a", 1000, [])
    summary = store.summary(now_ms=2000)
    assert summary["people"] == 0
    assert summary["protected_rate"] is None


def test_summary_hourly_buckets_cover_last_12_hours():
    store = EventStore()
    now = 100 * HOUR_MS + 5 * MINUTE_MS
    store.add("a", now, [person(True, True)])
    store.add("a", now - HOUR_MS, [person(False, True)])
    store.add("a", now - 13 * HOUR_MS, [person(True, True)])
    hourly = store.summary(now_ms=now)["hourly"]
    assert len(hourly) == 12
    assert hourly[-1] == {"hour_start": 100 * HOUR_MS, "checks": 1, "violations": 0}
    assert hourly[-2] == {"hour_start": 99 * HOUR_MS, "checks": 1, "violations": 1}
    assert sum(h["checks"] for h in hourly) == 2


def test_summary_cameras_live_or_stale():
    store = EventStore(live_seconds=15)
    now = 10 * HOUR_MS
    store.add("b", now - 5000, [])
    store.add("b", now - 8000, [])
    store.add("a", now - MINUTE_MS, [])
    cameras = store.summary(now_ms=now)["cameras"]
    assert cameras == [
        {"id": "a", "last_seen": now - MINUTE_MS, "checks": 1, "live": False},
        {"id": "b", "last_seen": now - 5000, "checks": 2, "live": True},
    ]
