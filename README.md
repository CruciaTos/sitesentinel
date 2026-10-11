# SiteSentinel

Watches a camera feed and tells you when someone on site isn't wearing a hard hat or a vest.

- **Detector** reads a video file or USB camera, finds people, hard hats and vests, and draws a live overlay.
- **Backend** collects each check, serves a dashboard at `http://localhost:8080`, and sends alerts to the console or a Webex space.

## Run it

Start the backend first.

```sh
cd backend
pip3 install -r requirements.txt
python3 app.py
```

Or with Docker:

```sh
cd backend
docker compose up --build
```

Then start the detector in a second terminal. It needs Python 3 with TensorFlow 1.12 or newer.

```sh
cd inference
pip3 install -r requirements.txt
python3 run_detector.py --model_dir=model --video_file_name=demo_video.mp4 --show_video_window=1 --camera_id=gate-1
```

| Option | Meaning |
| --- | --- |
| `--model_dir` | Folder containing `frozen_inference_graph.pb` |
| `--video_file_name` | A video file, or a camera such as `/dev/video0` (list them with `ls /dev/video*`) |
| `--show_video_window` | `1` shows the preview window, `0` hides it. Press `q` in the window to stop. |
| `--camera_id` | A name for this camera. It appears on the dashboard and in alerts. |

Open `http://localhost:8080` to watch checks arrive.

## Dashboard

- **Fully protected** shows the share of people seen in the last 15 minutes who wore both a hard hat and a vest.
- **Incident log** lists each check with a photo. Select a photo to enlarge it. Switch between violations and all checks.
- **Last 12 hours** shows checks per hour, with violations highlighted.
- **Cameras** shows which cameras are reporting.

## Alerts

By default alerts print to the backend's console. To post to a Webex space instead:

1. Create a bot at https://developer.webex.com and copy its token.
2. Add the bot to a space and get the room ID from https://developer.webex.com/docs/api/v1/rooms/list-rooms.
3. Set `PROVIDER=spark`, `SPARK_TOKEN` and `SPARK_ROOM_ID` before starting the backend.

To reduce repeat alerts, set `ALERT_COOLDOWN_SECONDS` (per camera). To add another channel, subclass `notification.Provider`.

## Backend settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `PORT` | `8080` | Port to listen on |
| `PROVIDER` | `console` | `console`, `spark` or anything else for no alerts |
| `WINDOW_MINUTES` | `15` | Time window for the protected percentage |
| `MAX_EVENTS` | `200` | Checks kept in memory (history resets when the backend restarts) |
| `ALERT_COOLDOWN_SECONDS` | `0` | Minimum gap between alerts per camera |

Detector settings (confidence threshold, resolution, send interval) are in `inference/config.py` and can be overridden with `PPE_*` environment variables.

## API

`POST /v1/detections` takes the detector's JSON (`cameraId`, `timestamp` in milliseconds, `persons`, optional base64 `image`). The dashboard reads `GET /api/summary`, `GET /api/events` and `GET /api/events/<id>/image`.

## Model

An SSD MobileNet v1 detector, initialised from COCO weights and trained to find hard hats, vests and people. The model file and demo video come from the original project. Training data isn't included.

## Origin and license

SiteSentinel is derived from [CiscoDevNet/ppe-detection](https://github.com/CiscoDevNet/ppe-detection) and is licensed under the Apache License 2.0. See `LICENSE` and `NOTICE` for the original copyright and a list of changes.
