import base64
import math

import pytest
from prometheus_client import REGISTRY

# Boxes are [ymin, xmin, ymax, xmax]. With the test model in conftest.py, an item is worn when
# it lies inside the person's box where that item is normally worn.
PERSON = [0.2, 0.2, 0.8, 0.4]
HAT_ON_HEAD = [0.2, 0.25, 0.29, 0.35]
VEST_ON_TORSO = [0.35, 0.22, 0.56, 0.38]
HAT_AT_WAIST = [0.5, 0.2, 0.58, 0.28]

METRIC_NAMES = [
    "sitesentinel_checks_total",
    "sitesentinel_people_total",
    "sitesentinel_violations_total",
    "sitesentinel_alerts_sent_total",
    "sitesentinel_protected_ratio",
    "sitesentinel_model_predictions_total",
    "sitesentinel_http_requests_total",
    "sitesentinel_http_request_duration_seconds",
    "sitesentinel_app_info",
    "sitesentinel_model_info",
]


def sample(name, **labels):
    """Current value of one metric series (0 if it doesn't exist yet)."""
    return REGISTRY.get_sample_value(name, labels) or 0.0


def detection(camera="gate-1", persons=None, **extra):
    body = {"cameraId": camera, "persons": persons if persons is not None else [{"hardhat": True, "vest": True}]}
    body.update(extra)
    return body


def frame(camera="gate-1", persons=None, hardhats=None, vests=None, **extra):
    body = {"cameraId": camera, "persons": persons if persons is not None else [PERSON]}
    if hardhats is not None:
        body["hardhats"] = hardhats
    if vests is not None:
        body["vests"] = vests
    body.update(extra)
    return body


# ---- health, version, dashboard ----

def test_healthz(client, backend, test_model):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok", "version": backend.config["app_version"], "model_version": "test-1"}


def test_version(client, backend, test_model):
    body = client.get("/api/version").get_json()
    assert body["app_version"] == backend.config["app_version"]
    assert body["model"]["version"] == "test-1"
    assert body["model"]["algorithm"] == "logistic_regression"
    assert body["model"]["metrics"] == {"overall": {"accuracy": 1.0, "f1": 1.0}}


def test_dashboard_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "SiteSentinel" in html
    assert 'id="versions"' in html
    assert "python simulator.py" in html


# ---- POST /v1/detections ----

def test_detection_is_recorded(client):
    resp = client.post("/v1/detections", json=detection(
        persons=[{"hardhat": True, "vest": True}, {"hardhat": False, "vest": True}], timestamp=1760000000000))
    assert resp.status_code == 201
    event = resp.get_json()
    assert event["camera"] == "gate-1"
    assert event["ts"] == 1760000000000
    assert (event["people"], event["protected"], event["no_hardhat"]) == (2, 1, 1)
    assert event["status"] == "violation"
    assert client.get("/api/events").get_json() == [event]


def test_detection_without_timestamp_uses_now(client):
    event = client.post("/v1/detections", json=detection()).get_json()
    assert event["ts"] > 1700000000000
    assert event["status"] == "clear"


@pytest.mark.parametrize("body", [
    None,
    ["not", "an", "object"],
    {"persons": []},
    {"cameraId": "", "persons": []},
    {"cameraId": 7, "persons": []},
    {"cameraId": "gate-1"},
    {"cameraId": "gate-1", "persons": {"hardhat": True}},
    {"cameraId": "gate-1", "persons": [{"hardhat": True}]},
    {"cameraId": "gate-1", "persons": ["helmet"]},
    {"cameraId": "gate-1", "persons": [], "timestamp": "yesterday"},
    {"cameraId": "gate-1", "persons": [], "image": "raw-bytes"},
])
def test_detection_rejects_bad_input(client, backend, body):
    if body is None:
        resp = client.post("/v1/detections", data="not json", content_type="text/plain")
    else:
        resp = client.post("/v1/detections", json=body)
    assert resp.status_code == 400
    assert resp.get_json()["message"]
    assert backend.store.events() == []


# ---- POST /v1/frames ----

def test_frame_with_hat_and_vest_is_clear(client, test_model):
    resp = client.post("/v1/frames", json=frame(hardhats=[HAT_ON_HEAD], vests=[VEST_ON_TORSO]))
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["status"] == "clear"
    assert body["persons"] == [{"hardhat": True, "vest": True}]
    assert body["model_version"] == "test-1"
    assert (body["people"], body["protected"]) == (1, 1)


def test_frame_uses_the_model_not_just_overlap(client, test_model):
    # The hat lies inside the person's box (the old overlap rule says "worn") but sits at the waist.
    body = client.post("/v1/frames", json=frame(hardhats=[HAT_AT_WAIST], vests=[VEST_ON_TORSO])).get_json()
    assert body["persons"] == [{"hardhat": False, "vest": True}]
    assert body["status"] == "violation"
    assert body["no_hardhat"] == 1


def test_frame_items_default_to_none(client, test_model):
    body = client.post("/v1/frames", json=frame(persons=[PERSON, [0.1, 0.6, 0.9, 0.8]])).get_json()
    assert body["persons"] == [{"hardhat": False, "vest": False}] * 2
    assert body["no_both"] == 2


def test_frame_stored_event_has_no_model_fields(client, test_model):
    image = base64.b64encode(b"jpeg").decode()
    body = client.post("/v1/frames", json=frame(hardhats=[HAT_ON_HEAD], image={"raw": image})).get_json()
    stored = client.get("/api/events").get_json()
    assert len(stored) == 1
    assert "persons" not in stored[0] and "model_version" not in stored[0]
    assert stored[0]["id"] == body["id"]
    assert stored[0]["has_image"] is True


@pytest.mark.parametrize("body", [
    {"persons": [PERSON]},
    {"cameraId": "gate-1"},
    {"cameraId": "gate-1", "persons": {"box": PERSON}},
    {"cameraId": "gate-1", "persons": [[0.1, 0.2, 0.3]]},
    {"cameraId": "gate-1", "persons": [[0.1, 0.2, 0.3, 0.4, 0.5]]},
    {"cameraId": "gate-1", "persons": [[0.1, "0.2", 0.3, 0.4]]},
    {"cameraId": "gate-1", "persons": [[0.1, True, 0.3, 0.4]]},
    {"cameraId": "gate-1", "persons": [0.1]},
    {"cameraId": "gate-1", "persons": [{"ymin": 0.1}]},
    {"cameraId": "gate-1", "persons": [PERSON], "hardhats": HAT_ON_HEAD},
    {"cameraId": "gate-1", "persons": [PERSON], "hardhats": "none"},
    {"cameraId": "gate-1", "persons": [PERSON], "vests": [[0.1, 0.2, None, 0.4]]},
    {"cameraId": "gate-1", "persons": [PERSON], "timestamp": [1]},
])
def test_frame_rejects_bad_input(client, backend, test_model, body):
    resp = client.post("/v1/frames", json=body)
    assert resp.status_code == 400
    assert resp.get_json()["message"]
    assert backend.store.events() == []


def test_frame_rejects_non_json(client):
    resp = client.post("/v1/frames", data="persons", content_type="text/plain")
    assert resp.status_code == 400


# ---- dashboard API ----

def test_events_filter_and_limit(client):
    client.post("/v1/detections", json=detection(timestamp=1000))
    client.post("/v1/detections", json=detection(timestamp=3000, persons=[{"hardhat": False, "vest": False}]))
    client.post("/v1/detections", json=detection(timestamp=2000))
    assert [e["ts"] for e in client.get("/api/events").get_json()] == [3000, 2000, 1000]
    assert [e["ts"] for e in client.get("/api/events?violations=1").get_json()] == [3000]
    assert [e["ts"] for e in client.get("/api/events?violations=true").get_json()] == [3000]
    assert [e["ts"] for e in client.get("/api/events?limit=2").get_json()] == [3000, 2000]
    assert len(client.get("/api/events?limit=0").get_json()) == 1


def test_event_image(client):
    raw = b"\xff\xd8 fake jpeg"
    event = client.post("/v1/detections", json=detection(image={"raw": base64.b64encode(raw).decode()})).get_json()
    assert event["has_image"] is True
    resp = client.get("/api/events/{}/image".format(event["id"]))
    assert resp.status_code == 200
    assert resp.mimetype == "image/jpeg"
    assert resp.data == raw


def test_event_image_missing(client):
    assert client.get("/api/events/nope/image").status_code == 404
    event = client.post("/v1/detections", json=detection()).get_json()
    assert client.get("/api/events/{}/image".format(event["id"])).status_code == 404


def test_summary(client):
    client.post("/v1/detections", json=detection(
        camera="crane-3", persons=[{"hardhat": True, "vest": True}, {"hardhat": True, "vest": False}]))
    summary = client.get("/api/summary").get_json()
    assert (summary["people"], summary["protected"], summary["protected_rate"]) == (2, 1, 50)
    assert summary["no_vest"] == 1
    assert [c["id"] for c in summary["cameras"]] == ["crane-3"]
    assert summary["cameras"][0]["live"] is True


# ---- metrics ----

def test_metrics_exposes_every_metric(client, test_model):
    client.post("/v1/frames", json=frame(camera="metrics-all", hardhats=[HAT_ON_HEAD]))
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.mimetype == "text/plain"
    text = resp.get_data(as_text=True)
    for name in METRIC_NAMES:
        assert name in text, name


def test_metrics_count_checks_people_and_violations(client):
    cam = "metrics-checks"
    before_violation = sample("sitesentinel_checks_total", camera=cam, status="violation")
    before_clear = sample("sitesentinel_checks_total", camera=cam, status="clear")
    persons = [{"hardhat": True, "vest": True}, {"hardhat": False, "vest": True}, {"hardhat": False, "vest": False}]
    client.post("/v1/detections", json=detection(camera=cam, persons=persons))
    client.post("/v1/detections", json=detection(camera=cam))
    assert sample("sitesentinel_checks_total", camera=cam, status="violation") == before_violation + 1
    assert sample("sitesentinel_checks_total", camera=cam, status="clear") == before_clear + 1
    assert sample("sitesentinel_people_total", camera=cam) == 4
    assert sample("sitesentinel_violations_total", camera=cam, kind="no_hardhat") == 1
    assert sample("sitesentinel_violations_total", camera=cam, kind="no_both") == 1
    # Series exist (at zero) as soon as a camera reports.
    assert REGISTRY.get_sample_value("sitesentinel_violations_total", {"camera": cam, "kind": "no_vest"}) == 0


def test_metrics_count_model_predictions(client, test_model):
    labels = [("hardhat", "worn"), ("hardhat", "not_worn"), ("vest", "worn"), ("vest", "not_worn")]
    before = {k: sample("sitesentinel_model_predictions_total", item=k[0], result=k[1]) for k in labels}
    client.post("/v1/frames", json=frame(camera="metrics-model", persons=[PERSON, PERSON],
                                         hardhats=[HAT_ON_HEAD], vests=[]))
    after = {k: sample("sitesentinel_model_predictions_total", item=k[0], result=k[1]) for k in labels}
    assert {k: after[k] - before[k] for k in labels} == {
        ("hardhat", "worn"): 2, ("hardhat", "not_worn"): 0, ("vest", "worn"): 0, ("vest", "not_worn"): 2,
    }


def test_metrics_count_http_requests(client):
    def requests(endpoint, status):
        return sample("sitesentinel_http_requests_total", method="GET", endpoint=endpoint, status=status)

    healthz, missing = requests("/healthz", "200"), requests("unmatched", "404")
    timed = sample("sitesentinel_http_request_duration_seconds_count", endpoint="/healthz")
    image = requests("/api/events/<event_id>/image", "404")
    client.get("/healthz")
    client.get("/no-such-page")
    client.get("/api/events/abc/image")
    client.get("/metrics")
    client.get("/static/app.css")
    assert requests("/healthz", "200") == healthz + 1
    assert requests("unmatched", "404") == missing + 1
    assert requests("/api/events/<event_id>/image", "404") == image + 1
    assert sample("sitesentinel_http_request_duration_seconds_count", endpoint="/healthz") == timed + 1
    assert requests("/metrics", "200") == 0
    assert requests("/static/<path:filename>", "200") == 0


def test_protected_ratio_follows_the_window(client):
    assert math.isnan(REGISTRY.get_sample_value("sitesentinel_protected_ratio"))
    client.post("/v1/detections", json=detection(
        persons=[{"hardhat": True, "vest": True}] * 3 + [{"hardhat": False, "vest": True}]))
    assert REGISTRY.get_sample_value("sitesentinel_protected_ratio") == 0.75


def test_info_metrics(backend):
    assert REGISTRY.get_sample_value("sitesentinel_app_info", {"version": backend.config["app_version"]}) == 1
    model = backend.model
    assert REGISTRY.get_sample_value(
        "sitesentinel_model_info", {"version": str(model.version), "algorithm": str(model.algorithm)}) == 1


# ---- alerts ----

def violation(camera):
    return detection(camera=camera, persons=[{"hardhat": False, "vest": True}])


def test_every_violation_alerts_without_cooldown(client, notifier):
    cam = "alerts-no-cooldown"
    before = sample("sitesentinel_alerts_sent_total", camera=cam)
    client.post("/v1/detections", json=violation(cam))
    client.post("/v1/detections", json=violation(cam))
    client.post("/v1/detections", json=detection(camera=cam))
    assert len(notifier.messages) == 2
    assert cam in notifier.messages[0]
    assert "Missing hard hat: 1" in notifier.messages[0]
    assert sample("sitesentinel_alerts_sent_total", camera=cam) == before + 2


def test_alert_cooldown_is_per_camera(client, backend, notifier, monkeypatch):
    monkeypatch.setitem(backend.config, "alert_cooldown_seconds", 60)
    first, second = "alerts-cooldown-1", "alerts-cooldown-2"
    before = sample("sitesentinel_alerts_sent_total", camera=first)
    client.post("/v1/detections", json=violation(first))
    client.post("/v1/detections", json=violation(first))
    client.post("/v1/detections", json=violation(second))
    assert len(notifier.messages) == 2
    assert first in notifier.messages[0] and second in notifier.messages[1]
    assert sample("sitesentinel_alerts_sent_total", camera=first) == before + 1


def test_failed_alert_is_not_counted(client, notifier):
    cam = "alerts-failed"
    notifier.result = False
    client.post("/v1/detections", json=violation(cam))
    assert len(notifier.messages) == 1
    assert sample("sitesentinel_alerts_sent_total", camera=cam) == 0
