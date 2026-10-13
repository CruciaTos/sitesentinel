"""SiteSentinel backend: receives detections, serves the dashboard and metrics, sends alerts."""
import math
import time
from datetime import datetime, timezone

from flask import Flask, Response, abort, jsonify, render_template, request
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

import metrics
import ppe_model
from config import config
from notification.factory import NotificationFactory
from store import EventStore

app = Flask(__name__)
store = EventStore(
    max_events=config["max_events"],
    window_minutes=config["window_minutes"],
)
notifier = NotificationFactory.instance(config)
model = ppe_model.load(config["model_path"])
print("[Info] PPE model", model.version, "({})".format(model.algorithm))
_last_alert = {}

metrics.install(app)
metrics.set_info(config["app_version"], model)
metrics.track_store(lambda: store)

ALERT_TEMPLATE = (
    "### SiteSentinel alert\n"
    "**{camera}** at {time}\n"
    "> {people} on camera. Missing hard hat: {no_hardhat}. "
    "Missing vest: {no_vest}. Missing both: {no_both}.\n"
)


class HttpError(Exception):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@app.errorhandler(HttpError)
def handle_http_error(err):
    print("[Error]", err.message)
    return jsonify({"message": err.message}), err.status_code


def _format_alert(event):
    when = datetime.fromtimestamp(event["ts"] / 1000, tz=timezone.utc)
    return ALERT_TEMPLATE.format(
        camera=event["camera"],
        time=when.strftime("%Y-%m-%d %H:%M:%S UTC"),
        people=event["people"],
        no_hardhat=event["no_hardhat"],
        no_vest=event["no_vest"],
        no_both=event["no_both"],
    )


def _should_alert(camera_id):
    cooldown = config["alert_cooldown_seconds"]
    now = time.time()
    if cooldown and now - _last_alert.get(camera_id, 0) < cooldown:
        return False
    _last_alert[camera_id] = now
    return True


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _read_check():
    """The fields shared by /v1/detections and /v1/frames: (body, camera id, timestamp ms, base64 image)."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise HttpError("Send a JSON body.")
    camera_id = body.get("cameraId")
    if not camera_id or not isinstance(camera_id, str):
        raise HttpError("cameraId is required.")
    timestamp = body.get("timestamp") or int(time.time() * 1000)
    if not _is_number(timestamp):
        raise HttpError("timestamp must be a number of milliseconds.")
    image = body.get("image") or {}
    if not isinstance(image, dict):
        raise HttpError("image must be an object with a base64 raw field.")
    return body, camera_id, timestamp, image.get("raw")


def _read_boxes(body, key, required=False):
    boxes = body.get(key)
    if boxes is None and not required:
        return []
    if not isinstance(boxes, list):
        raise HttpError("{} must be a list of boxes.".format(key))
    for box in boxes:
        if not (isinstance(box, list) and len(box) == 4 and all(_is_number(v) for v in box)):
            raise HttpError("Each box in {} must be [ymin, xmin, ymax, xmax].".format(key))
    return [[float(v) for v in box] for box in boxes]


def _record(camera_id, timestamp, persons, image_b64):
    """Store one check, update the metrics and send an alert if someone is unprotected."""
    event = store.add(camera_id, timestamp, persons, image_b64)
    metrics.record_check(event)

    if event["status"] == "violation":
        print("[Warn]", camera_id, "people without full protection")
        if _should_alert(camera_id):
            # Console.send returns None; Spark.send returns False when Webex rejects the message.
            if notifier.send(_format_alert(event)) is not False:
                metrics.record_alert(camera_id)
    else:
        print("[Info]", camera_id, "everyone protected")
    return event


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "version": config["app_version"], "model_version": str(model.version)})


@app.route("/api/version")
def api_version():
    return jsonify({"app_version": config["app_version"], "model": model.info()})


@app.route("/metrics")
def prometheus_metrics():
    return Response(generate_latest(), content_type=CONTENT_TYPE_LATEST)


# The detector posts one JSON document per check:
# {
#   "cameraId": "camera1",
#   "timestamp": 1558506692000,          (milliseconds)
#   "persons": [{"hardhat": true, "vest": false}, ...],
#   "image": {"height": 360, "width": 640, "format": "jpeg", "raw": "<base64>"}
# }
@app.route("/v1/detections", methods=["POST"])
def create_detection():
    body, camera_id, timestamp, image_b64 = _read_check()
    persons = body.get("persons")
    if not isinstance(persons, list):
        raise HttpError("persons must be a list.")
    try:
        persons = [{"hardhat": bool(p["hardhat"]), "vest": bool(p["vest"])} for p in persons]
    except (KeyError, TypeError):
        raise HttpError("Each person needs hardhat and vest values.")

    event = _record(camera_id, timestamp, persons, image_b64)
    return jsonify(event), 201


# The simulator (or a detector that leaves the matching to the backend) posts raw boxes,
# each a normalized [ymin, xmin, ymax, xmax]; the PPE model decides who wears what:
# {"cameraId": "gate-1", "timestamp": 1760000000000,
#  "persons": [[...], ...], "hardhats": [[...], ...], "vests": [[...], ...], "image": {"raw": "<base64>"}}
@app.route("/v1/frames", methods=["POST"])
def create_frame():
    body, camera_id, timestamp, image_b64 = _read_check()
    person_boxes = _read_boxes(body, "persons", required=True)
    hardhats = _read_boxes(body, "hardhats")
    vests = _read_boxes(body, "vests")

    persons = ppe_model.assess_people(model, person_boxes, hardhats, vests)
    metrics.record_predictions(persons)
    event = _record(camera_id, timestamp, persons, image_b64)
    return jsonify(dict(event, model_version=str(model.version), persons=persons)), 201


@app.route("/api/summary")
def api_summary():
    return jsonify(store.summary())


@app.route("/api/events")
def api_events():
    limit = max(1, min(request.args.get("limit", 50, type=int), 200))
    only_violations = request.args.get("violations") in ("1", "true")
    return jsonify(store.events(limit=limit, violations_only=only_violations))


@app.route("/api/events/<event_id>/image")
def api_event_image(event_id):
    data = store.image(event_id)
    if data is None:
        abort(404)
    return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "max-age=3600"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config["port"])
