"""Detector settings. Every value can be overridden with an environment variable."""
import os

# The SiteSentinel backend. Frames (boxes, judged by the backend's trained model) go to /v1/frames;
# --report detections sends the detector's own decisions to /v1/detections instead.
backend_url = os.getenv("PPE_BACKEND_URL", "http://localhost:8080").rstrip("/")
frames_api = os.getenv("PPE_FRAMES_URL", backend_url + "/v1/frames")
detection_api = os.getenv("PPE_DETECTION_URL", backend_url + "/v1/detections")

# Minimum time between messages to the backend, in milliseconds.
message_send_interval = int(os.getenv("PPE_MESSAGE_SEND_INTERVAL", 1000))

# Detections scoring below this are ignored.
object_confidence_threshold = float(os.getenv("PPE_OBJECT_CONFIDENCE_THRESHOLD", .5))

# Capture size. Supported: (640, 480), (1280, 720), (1920, 1080).
supported_video_resolution = [(640, 480), (1280, 720), (1920, 1080)]
capture_image_width = int(os.getenv("PPE_CAPTURE_IMAGE_WIDTH", 1280))
capture_image_height = int(os.getenv("PPE_CAPTURE_IMAGE_HEIGHT", 720))

# Preview window.
display_full_screen = os.getenv("PPE_DISPLAY_FULL_SCREEN", "False").lower() == "true"
display_window_width = int(os.getenv("PPE_DISPLAY_WINDOW_WIDTH", 1280))
display_window_height = int(os.getenv("PPE_DISPLAY_WINDOW_HEIGHT", 720))

# Size of the snapshot saved with each check.
storage_image_width = int(os.getenv("PPE_STORAGE_IMAGE_WIDTH", 640))
storage_image_height = int(os.getenv("PPE_STORAGE_IMAGE_HEIGHT", 360))
