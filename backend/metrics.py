"""Prometheus metrics for the backend, served at /metrics.

Counters are named without the _total suffix; prometheus_client adds it, so
Counter("sitesentinel_checks") is exposed as sitesentinel_checks_total.
"""
import time

from flask import g, request
from prometheus_client import Counter, Gauge, Histogram, Info, disable_created_metrics

# Keep the output short: no *_created series next to every counter.
disable_created_metrics()

VIOLATION_KINDS = ("no_hardhat", "no_vest", "no_both")

CHECKS = Counter("sitesentinel_checks", "Safety checks recorded.", ["camera", "status"])
PEOPLE = Counter("sitesentinel_people", "People seen by the cameras.", ["camera"])
VIOLATIONS = Counter("sitesentinel_violations", "People missing protective equipment.", ["camera", "kind"])
ALERTS_SENT = Counter("sitesentinel_alerts_sent", "Alert messages sent.", ["camera"])
PROTECTED_RATIO = Gauge(
    "sitesentinel_protected_ratio",
    "Share of people wearing a hard hat and a vest in the dashboard window (NaN when nobody was seen).",
)
MODEL_PREDICTIONS = Counter(
    "sitesentinel_model_predictions", "Per-person PPE decisions made by the model.", ["item", "result"]
)
HTTP_REQUESTS = Counter("sitesentinel_http_requests", "HTTP requests handled.", ["method", "endpoint", "status"])
HTTP_DURATION = Histogram("sitesentinel_http_request_duration_seconds", "HTTP request latency.", ["endpoint"])
APP_INFO = Info("sitesentinel_app", "Backend version.")
MODEL_INFO = Info("sitesentinel_model", "PPE association model in use.")

# Expose every model decision series from the start, at zero.
for _item in ("hardhat", "vest"):
    for _result in ("worn", "not_worn"):
        MODEL_PREDICTIONS.labels(_item, _result)


def set_info(app_version, model):
    APP_INFO.info({"version": str(app_version)})
    MODEL_INFO.info({"version": str(model.version), "algorithm": str(model.algorithm)})


def track_store(get_store):
    """Compute sitesentinel_protected_ratio from the store's summary at scrape time."""
    def ratio():
        summary = get_store().summary()
        if not summary["people"]:
            return float("nan")
        return summary["protected"] / summary["people"]
    PROTECTED_RATIO.set_function(ratio)


def record_check(event):
    """Count one stored check (an event from EventStore.add)."""
    camera = event["camera"]
    # Touch every series for this camera so rates start from zero instead of missing.
    for status in ("clear", "violation"):
        CHECKS.labels(camera, status)
    ALERTS_SENT.labels(camera)
    CHECKS.labels(camera, event["status"]).inc()
    PEOPLE.labels(camera).inc(event["people"])
    for kind in VIOLATION_KINDS:
        VIOLATIONS.labels(camera, kind).inc(event[kind])


def record_alert(camera):
    ALERTS_SENT.labels(camera).inc()


def record_predictions(persons):
    """Count the model's decisions; `persons` is a list of {"hardhat": bool, "vest": bool}."""
    for person in persons:
        for item in ("hardhat", "vest"):
            MODEL_PREDICTIONS.labels(item, "worn" if person[item] else "not_worn").inc()


def _tracked():
    return request.path != "/metrics" and not request.path.startswith("/static/")


def install(app):
    """Count requests and time them, except /metrics itself and static files."""
    @app.before_request
    def _start_timer():
        g.metrics_start = time.perf_counter()

    @app.after_request
    def _observe(response):
        start = g.pop("metrics_start", None)
        if start is not None and _tracked():
            endpoint = request.url_rule.rule if request.url_rule else "unmatched"
            HTTP_REQUESTS.labels(request.method, endpoint, str(response.status_code)).inc()
            HTTP_DURATION.labels(endpoint).observe(time.perf_counter() - start)
        return response
