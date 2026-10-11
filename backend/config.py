import os

config = {
    "port": int(os.getenv("PORT", 8080)),
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
