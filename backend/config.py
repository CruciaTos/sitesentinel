import os

import ppe_model

config = {
    "port": int(os.getenv("PORT", 8080)),
    # Shown on the dashboard, in /healthz and in the sitesentinel_app_info metric.
    "app_version": os.getenv("APP_VERSION", "dev"),
    # Trained PPE association model (written by ml/train.py).
    "model_path": os.getenv("MODEL_PATH", ppe_model.DEFAULT_PATH),
    # Where safety alerts go: "console" prints them, "spark" posts to a Webex space.
    "provider": os.getenv("PROVIDER", "console"),
    "spark": {
        "url": os.getenv("SPARK_URL", "https://api.ciscospark.com/v1"),
        "token": os.getenv("SPARK_TOKEN", "your token"),
        "room_id": os.getenv("SPARK_ROOM_ID", "your room id"),
    },
    "console": {
        "room_id": "console",
    },
    # Dashboard settings
    "max_events": int(os.getenv("MAX_EVENTS", 200)),
    "window_minutes": int(os.getenv("WINDOW_MINUTES", 15)),
    # Minimum seconds between alert messages for the same camera.
    "alert_cooldown_seconds": int(os.getenv("ALERT_COOLDOWN_SECONDS", 0)),
}
