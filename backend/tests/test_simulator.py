import io
import json
import random
import urllib.error

import pytest

import simulator

SEEDS = range(300)
HAT_BAND = (0.0, 0.2)
VEST_BAND = (0.25, 0.6)


def in_band(item, person, band):
    """Item centre is inside the person's box horizontally, and within `band` of its height from the top."""
    cy, cx = (item[0] + item[2]) / 2, (item[1] + item[3]) / 2
    rel_cy = (cy - person[0]) / (person[2] - person[0])
    return person[1] <= cx <= person[3] and band[0] <= rel_cy <= band[1]


def created(status=201, body=None):
    return status, body or {"status": "clear", "people": 0, "protected": 0, "model_version": "m-1"}


@pytest.mark.parametrize("violation_rate", [0.0, 0.3, 1.0])
def test_scenes_are_valid(violation_rate):
    counts = set()
    for seed in SEEDS:
        payload, truth = simulator.make_scene(random.Random(seed), violation_rate, "cam-1", now_ms=123)
        assert set(payload) == {"cameraId", "timestamp", "persons", "hardhats", "vests"}
        assert (payload["cameraId"], payload["timestamp"]) == ("cam-1", 123)
        assert len(truth) == len(payload["persons"]) <= 4
        counts.add(len(truth))
        for box in payload["persons"] + payload["hardhats"] + payload["vests"]:
            assert len(box) == 4 and all(isinstance(v, float) for v in box)
            assert 0.0 <= box[0] < box[2] <= 1.0
            assert 0.0 <= box[1] < box[3] <= 1.0
        json.dumps(payload)
    assert counts == {0, 1, 2, 3, 4}


def test_worn_items_sit_on_head_and_torso_and_missing_ones_do_not():
    for seed in SEEDS:
        payload, truth = simulator.make_scene(random.Random(seed), 0.5)
        for person, wears in zip(payload["persons"], truth):
            assert any(in_band(h, person, HAT_BAND) for h in payload["hardhats"]) == wears["hardhat"]
            assert any(in_band(v, person, VEST_BAND) for v in payload["vests"]) == wears["vest"]


def test_people_never_overlap():
    for seed in SEEDS:
        persons = simulator.make_scene(random.Random(seed), 0.3)[0]["persons"]
        spans = sorted((p[1], p[3]) for p in persons)
        assert all(a[1] < b[0] for a, b in zip(spans, spans[1:]))


def test_violation_rate_extremes():
    for seed in SEEDS:
        assert all(t["hardhat"] and t["vest"] for t in simulator.make_scene(random.Random(seed), 0.0)[1])
        assert not any(t["hardhat"] and t["vest"] for t in simulator.make_scene(random.Random(seed), 1.0)[1])


def test_missing_items_sometimes_show_up_as_distractors():
    extra = 0
    for seed in SEEDS:
        payload, truth = simulator.make_scene(random.Random(seed), 1.0)
        extra += len(payload["hardhats"]) - sum(t["hardhat"] for t in truth)
        extra += len(payload["vests"]) - sum(t["vest"] for t in truth)
    assert extra > 50


def test_same_seed_same_scene():
    scene = simulator.make_scene(random.Random(5), 0.3, now_ms=1)
    assert scene == simulator.make_scene(random.Random(5), 0.3, now_ms=1)


def test_run_sends_count_frames_round_robin():
    calls, sleeps, lines = [], [], []

    def fake_post(url, payload):
        calls.append((url, payload["cameraId"]))
        return created()

    sent = simulator.run("http://backend:8080/", ["a", "b"], 2.0, 0.3, count=5, rng=random.Random(1),
                         post_fn=fake_post, sleep=sleeps.append, log=lines.append)
    assert sent == 5
    assert [cam for _, cam in calls] == ["a", "b", "a", "b", "a"]
    assert {url for url, _ in calls} == {"http://backend:8080/v1/frames"}
    assert sleeps == [2.0] * 4
    assert len(lines) == 5 and "-> 201 clear (model m-1" in lines[0]


def test_run_retries_with_backoff_until_backend_is_up():
    attempts, sleeps, lines = [], [], []

    def flaky_post(url, payload):
        attempts.append(payload)
        if len(attempts) < 5:
            raise ConnectionRefusedError("refused")
        return created()

    sent = simulator.run("http://backend:8080", ["a"], 2.0, 0.3, count=1, rng=random.Random(1),
                         post_fn=flaky_post, sleep=sleeps.append, log=lines.append)
    assert sent == 1
    assert len(attempts) == 5 and all(p == attempts[0] for p in attempts)
    assert sleeps == [1, 2, 4, 8]
    assert sum("not reachable" in line for line in lines) == 4


def test_run_reports_rejected_frames():
    lines = []
    sent = simulator.run("http://backend:8080", ["a"], 0, 0.3, count=1, rng=random.Random(1),
                         post_fn=lambda url, payload: (400, {"message": "persons must be a list of boxes."}),
                         sleep=lambda s: None, log=lines.append)
    assert sent == 1
    assert "-> 400 persons must be a list of boxes." in lines[0]


def test_main_with_count(monkeypatch, capsys):
    monkeypatch.setenv("CAMERA_IDS", "x, y")
    monkeypatch.setenv("INTERVAL_SECONDS", "0")
    monkeypatch.setenv("SEED", "7")
    monkeypatch.setenv("BACKEND_URL", "http://not-used:1")
    calls = []

    def fake_post(url, payload):
        calls.append((url, payload["cameraId"]))
        return created()

    monkeypatch.setattr(simulator, "post", fake_post)
    monkeypatch.setattr(simulator.signal, "signal", lambda signum, handler: None)
    assert simulator.main(["--count", "3", "--url", "http://sim-test:9"]) == 0
    assert calls == [("http://sim-test:9/v1/frames", c) for c in ["x", "y", "x"]]
    assert capsys.readouterr().out.count("-> 201") == 3


class FakeResponse(io.BytesIO):
    status = 201

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_post_returns_status_and_json(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen.update(url=req.full_url, method=req.get_method(), body=json.loads(req.data),
                    type=req.get_header("Content-type"))
        return FakeResponse(b'{"status": "clear"}')

    monkeypatch.setattr(simulator.urllib.request, "urlopen", fake_urlopen)
    assert simulator.post("http://b/v1/frames", {"cameraId": "a"}) == (201, {"status": "clear"})
    assert seen == {"url": "http://b/v1/frames", "method": "POST", "body": {"cameraId": "a"},
                    "type": "application/json"}


def test_post_returns_http_errors(monkeypatch):
    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(b'{"message": "no"}'))

    monkeypatch.setattr(simulator.urllib.request, "urlopen", fake_urlopen)
    assert simulator.post("http://b/v1/frames", {}) == (400, {"message": "no"})


def test_post_raises_when_unreachable(monkeypatch):
    def fake_urlopen(req, timeout):
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    monkeypatch.setattr(simulator.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(OSError):
        simulator.post("http://b/v1/frames", {})
