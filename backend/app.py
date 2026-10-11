"""SiteSentinel backend: receives detections, serves the dashboard, sends alerts."""
import time
from datetime import datetime, timezone

from flask import Flask, Response, abort, jsonify, render_template, request

from config import config
from notification.factory import NotificationFactory
from store import EventStore

app = Flask(__name__)
store = EventStore(
    max_events=config["max_events"],
    window_minutes=config["window_minutes"],
)
notifier = NotificationFactory.instance(config)
_last_alert = {}

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


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


# The detector posts one JSON document per check:
# {
#   "cameraId": "camera1",
#   "timestamp": 1558506692000,          (milliseconds)
#   "persons": [{"hardhat": true, "vest": false}, ...],
#   "image": {"height": 360, "width": 640, "format": "jpeg", "raw": "<base64>"}
# }
@app.route("/v1/detections", methods=["POST"])
def create_detection():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise HttpError("Send a JSON body.")
    camera_id = body.get("cameraId")
    if not camera_id:
        raise HttpError("cameraId is required.")
    persons = body.get("persons")
    if not isinstance(persons, list):
        raise HttpError("persons must be a list.")
    try:
        persons = [{"hardhat": bool(p["hardhat"]), "vest": bool(p["vest"])} for p in persons]
    except (KeyError, TypeError):
        raise HttpError("Each person needs hardhat and vest values.")

    timestamp = body.get("timestamp") or int(time.time() * 1000)
    image = body.get("image") or {}
    event = store.add(camera_id, timestamp, persons, image.get("raw"))

    if event["status"] == "violation":
        print("[Warn]", camera_id, "people without full protection")
        if _should_alert(camera_id):
            notifier.send(_format_alert(event))
    else:
        print("[Info]", camera_id, "everyone protected")

    return jsonify(event), 201


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
