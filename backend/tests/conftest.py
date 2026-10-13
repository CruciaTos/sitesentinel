import json

import pytest

import ppe_model
from store import EventStore

# A tiny hand-made logistic model, so API tests don't depend on the trained model/ppe_model.json:
# an item counts as worn when it lies inside the person's box (share_inside) close to where that
# item is normally worn (cy_dist). All other weights are zero.
WEIGHTS = {"share_inside": 10.0, "cy_dist": -20.0}
TEST_MODEL = {
    "name": "ppe-association",
    "version": "test-1",
    "algorithm": "logistic_regression",
    "trained_at": "2026-01-01T00:00:00Z",
    "features": ppe_model.FEATURES,
    "threshold": 0.5,
    "models": {
        item: {
            "mean": [0.0] * len(ppe_model.FEATURES),
            "scale": [1.0] * len(ppe_model.FEATURES),
            "coef": [WEIGHTS.get(name, 0.0) for name in ppe_model.FEATURES],
            "intercept": -4.0,
        }
        for item in ppe_model.ITEM_TYPES
    },
    "metrics": {"overall": {"accuracy": 1.0, "f1": 1.0}},
}


class FakeNotifier:
    """Records alert messages instead of printing or posting them."""

    def __init__(self, result=None):
        self.messages = []
        self.result = result

    def send(self, msg):
        self.messages.append(msg)
        return self.result


@pytest.fixture
def backend():
    """The app module (imported lazily so store/simulator tests don't need it)."""
    import app
    return app


@pytest.fixture
def notifier(backend, monkeypatch):
    fake = FakeNotifier()
    monkeypatch.setattr(backend, "notifier", fake)
    return fake


@pytest.fixture
def client(backend, notifier, monkeypatch):
    """A test client with an empty event store and no alert cooldown."""
    monkeypatch.setattr(backend, "store", EventStore(max_events=50, window_minutes=15))
    monkeypatch.setattr(backend, "_last_alert", {})
    monkeypatch.setitem(backend.config, "alert_cooldown_seconds", 0)
    monkeypatch.setitem(backend.app.config, "TESTING", True)
    return backend.app.test_client()


@pytest.fixture
def test_model(backend, tmp_path, monkeypatch):
    """Serve /v1/frames with TEST_MODEL, loaded through ppe_model.load like the real one."""
    path = tmp_path / "ppe_model.json"
    path.write_text(json.dumps(TEST_MODEL))
    model = ppe_model.load(str(path))
    assert isinstance(model, ppe_model.LogisticModel)
    monkeypatch.setattr(backend, "model", model)
    return model
