# SiteSentinel

Watches a camera feed and tells you when someone on site isn't wearing a hard hat or a vest, delivered
through an end-to-end DevOps pipeline: every commit is linted, tested, used to retrain and validate the
ML model, packaged as a Docker image, pushed to a registry, deployed, smoke-tested (with automatic
rollback) and monitored.

![Pipeline architecture](docs/architecture.png)

- **Backend** (`backend/`) collects each check, serves a dashboard at `http://localhost:8080`, exposes
  Prometheus metrics and sends alerts to the console or a Webex space.
- **PPE model** (`ml/`, `backend/ppe_model.py`) decides whether a detected hard hat or vest is worn by a
  given person. It is retrained and quality-gated on every pipeline run. See [ml/README.md](ml/README.md).
- **Camera simulator** (`backend/simulator.py`) sends realistic frames so the whole system runs without a
  camera or GPU.
- **Detector** (`inference/`) is the original TensorFlow detector for a real video file or USB camera.

## Run it

You need Docker Desktop. From the repo root:

```sh
docker compose up -d --build
```

| What | Where |
| --- | --- |
| SiteSentinel dashboard | http://localhost:8080 |
| Grafana (admin / admin) | http://localhost:3000 |
| Prometheus and its alerts | http://localhost:9090/alerts |

The simulator starts posting frames as soon as the backend is healthy. With its default settings about
30% of workers miss some PPE, so the `LowPPECompliance` alert fires after about a minute: that is the demo.
Stop everything with `docker compose down`.

Without Docker (Python 3.12):

```sh
cd backend
pip install -r requirements.txt
python app.py                  # terminal 1
python simulator.py            # terminal 2
```

## The pipeline

| Stage | GitHub Actions (`.github/workflows/ci-cd.yml`) | Jenkins (`Jenkinsfile`) |
| --- | --- | --- |
| Lint & unit tests | flake8, pytest (+ coverage), promtool alert-rule tests | flake8, pytest |
| Train & validate model | `ml/train.py` with the quality gate | same |
| Build image | multi-stage `backend/Dockerfile`, tag `sha-<commit>` | same, plus the `test` build target |
| Registry | GitHub Container Registry (`ghcr.io/<owner>/sitesentinel-backend`) | local `registry:2` at `localhost:5000` |
| Deploy | Kubernetes cluster (kind) inside the runner, `scripts/k8s-deploy.sh` | Docker Compose on this machine, `scripts/deploy.sh` |
| Verify | `scripts/smoke-test.sh`; `kubectl rollout undo` on failure | `scripts/smoke-test.sh`; `scripts/rollback.sh` on failure |

Pull requests run everything up to the image build and deploy nothing. Pushes to `main`, the weekly
schedule (Monday 03:30 UTC, retraining) and "Run workflow" also push and deploy.

### GitHub Actions

Create a GitHub repository, then push:

```sh
git remote add origin https://github.com/<you>/sitesentinel.git
git push -u origin main
```

The workflow starts by itself (Actions tab). The image is pushed with the built-in `GITHUB_TOKEN`; no
secrets are needed. The `staging` environment appears on the repo page after the first deployment.

### Jenkins (local)

Jenkins builds the **committed** `main` branch of this folder.

```sh
docker compose -f jenkins/docker-compose.yml up -d --build
```

Open http://localhost:8081 (admin / admin) and open the `sitesentinel` job. It polls the repo every
2 minutes, or use **Build Now**. To demonstrate the automatic rollback, use **Build with Parameters**
and tick `SIMULATE_FAILED_DEPLOY`: the smoke test fails and the previous image is put back. Do this on a
new commit: the rollback restores the image that ran before, so redeploying the commit that is already running
has nothing to roll back to. The `latest` tag is only moved after a build passes its smoke test.
Stop Jenkins with `docker compose -f jenkins/docker-compose.yml down`.

### Kubernetes (local, optional)

Uses [kind](https://kind.sigs.k8s.io/). The node image is pinned to Kubernetes 1.34 because newer ones
refuse to run on Docker Desktop's cgroup v1 VM.

```sh
kind create cluster --name sitesentinel --image kindest/node:v1.34.8@sha256:02722c2dedddcfc00febf5d27fbeb9b7b2c14294c82109ff4a85d89ac9ba3256
docker build --build-arg APP_VERSION=v1 -t sitesentinel-backend:v1 backend
kind load docker-image sitesentinel-backend:v1 --name sitesentinel
bash scripts/k8s-deploy.sh sitesentinel-backend:v1
kubectl get all -n sitesentinel
```

### Deploy and roll back by hand (Docker Compose)

```sh
docker build --build-arg APP_VERSION=v2 -t sitesentinel-backend:v2 backend
bash scripts/deploy.sh sitesentinel-backend v2
bash scripts/smoke-test.sh http://localhost:8080 v2
bash scripts/rollback.sh                      # back to the image that ran before
```

## Monitoring

Prometheus scrapes `/metrics` every 5 seconds. Grafana opens on the "SiteSentinel – Safety & Service Health"
dashboard: compliance %, checks and violations per minute, model decisions, request rate, p95 latency,
5xx ratio, and the running app and model versions.

| Alert | Fires when |
| --- | --- |
| `SiteSentinelDown` | the backend cannot be scraped for 30 s (try `docker compose stop backend`) |
| `LowPPECompliance` | fewer than 80% of people are fully protected for 1 min |
| `NoChecksReceived` | the backend is up but no checks arrive for 2 min |
| `HighErrorRate` | more than 5% of requests fail with 5xx for 2 min |

The rules are unit-tested with `promtool test rules monitoring/prometheus/alert-rules.test.yml`.

## Tests

```sh
cd backend && pip install -r requirements-dev.txt && flake8 . && pytest
cd ml && pip install -r requirements.txt pytest flake8 && flake8 . && pytest
cd inference && pip install -r requirements-test.txt && flake8 . && pytest
```

## Dashboard

- **Fully protected** shows the share of people seen in the last 15 minutes who wore both a hard hat and a vest.
- **Incident log** lists each check, with a photo when the camera sent one. Select a photo to enlarge it.
  Switch between violations and all checks.
- **Last 12 hours** shows checks per hour, with violations highlighted.
- **Cameras** shows which cameras are reporting.
- The footer shows the running app version and model version.

## Alerts

By default alerts print to the backend's console. To post to a Webex space instead:

1. Create a bot at https://developer.webex.com and copy its token.
2. Add the bot to a space and get the room ID from https://developer.webex.com/docs/api/v1/rooms/list-rooms.
3. Set `PROVIDER=spark`, `SPARK_TOKEN` and `SPARK_ROOM_ID` before starting the backend.

To reduce repeat alerts, set `ALERT_COOLDOWN_SECONDS` (per camera). To add another channel, subclass `notification.Provider`.

## Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `PORT` | `8080` | Port to listen on (`python app.py` only; the container always uses 8080) |
| `PROVIDER` | `console` | `console`, `spark` or anything else for no alerts |
| `WINDOW_MINUTES` | `15` | Time window for the protected percentage |
| `MAX_EVENTS` | `200` | Checks kept in memory (history resets when the backend restarts) |
| `ALERT_COOLDOWN_SECONDS` | `0` | Minimum gap between alerts per camera |
| `APP_VERSION` | `dev` | Version reported by `/healthz`; the pipeline sets it to `sha-<commit>` |
| `MODEL_PATH` | `backend/model/ppe_model.json` | Trained model; the overlap rule is used if it is missing |

Simulator: `python simulator.py [--count N] [--url URL]`, plus `BACKEND_URL`, `CAMERA_IDS`
(default `gate-1,scaffold-2,crane-3`), `INTERVAL_SECONDS` (2), `VIOLATION_RATE` (0.3) and `SEED`.

## API

- `POST /v1/frames` takes boxes: `cameraId`, optional `timestamp` (ms), `persons`, `hardhats`, `vests`
  (each a list of `[ymin, xmin, ymax, xmax]`, normalized) and an optional base64 `image`. The model decides
  who wears what.
- `POST /v1/detections` takes the detector's decisions: `cameraId`, `timestamp`, `persons` with `hardhat` and
  `vest` booleans, optional `image`.
- `GET /healthz` (status, app version, model version), `GET /api/version`, `GET /metrics`.
- The dashboard reads `GET /api/summary`, `GET /api/events` and `GET /api/events/<id>/image`.

## Live cameras (detector)

The simulator stands in for cameras. To use real ones, run the TensorFlow detector (`inference/`) once per
camera. It reads a webcam, an IP camera stream or a video file, finds people, hard hats and vests, and
posts the boxes to `/v1/frames`, where the backend's trained model decides who is protected. The camera then
appears on the dashboard, in Grafana and in the alerts under its `--camera_id`.

```sh
sh get_assets.sh                       # detector model + demo video (once)
cd inference
pip install -r requirements.txt        # Python 3.12, TensorFlow 2.20
python run_detector.py --model_dir=model --source=demo_video.mp4 --camera_id=gate-1 --show_video_window=1
```

| `--source` | Example |
| --- | --- |
| Webcam (number) | `--source=0` |
| IP / CCTV camera (RTSP) | `--source=rtsp://user:password@192.168.1.20:554/stream1` |
| MJPEG / HTTP stream | `--source=http://192.168.1.21:8080/video` |
| Linux device | `--source=/dev/video0` |
| Video file (loops) | `--source=demo_video.mp4` |

| Option | Meaning |
| --- | --- |
| `--model_dir` | Folder containing `frozen_inference_graph.pb` |
| `--camera_id` | A name for this camera, shown on the dashboard and in alerts |
| `--show_video_window` | `1` shows the preview window with the overlay (press `q` to stop); default `0` |
| `--report` | `frames` (default): send boxes, the backend model decides. `detections`: send the detector's own overlap-rule decisions to `/v1/detections` |

- Live cameras reconnect by themselves (1 s, 2 s, 4 s ... up to 30 s) when the stream drops or is not
  reachable yet; video files loop.
- Several cameras: start one detector per camera, each with its own `--camera_id`.
- Backend on another machine: set `PPE_BACKEND_URL=http://<backend-host>:8080`.
- Turn the simulator off when real cameras are connected: `docker compose stop simulator`.
- Other settings (confidence threshold, resolution, send interval) are in `inference/config.py` and can be
  overridden with `PPE_*` environment variables.
- The camera and message code is unit-tested in both pipelines (`cd inference && pytest`, no TensorFlow needed).

The detector model is an SSD MobileNet v1, initialised from COCO weights and trained to find hard hats,
vests and people; it comes from the original project.

## Project layout

| Path | Contents |
| --- | --- |
| `backend/` | Flask app, metrics, PPE model loader, simulator, tests, Dockerfile |
| `ml/` | Training data generator, training script with quality gate, data, metrics report |
| `monitoring/` | Prometheus config and alert rules, Grafana provisioning and dashboard |
| `k8s/` | Kubernetes manifests |
| `scripts/` | Deploy, rollback, Kubernetes deploy and smoke-test scripts |
| `jenkins/`, `Jenkinsfile` | Local Jenkins with a registry, and its pipeline |
| `.github/workflows/` | GitHub Actions pipeline |
| `inference/` | TensorFlow detector |
| `docs/` | Architecture diagram |

## Origin and license

SiteSentinel is derived from [CiscoDevNet/ppe-detection](https://github.com/CiscoDevNet/ppe-detection) and is
licensed under the Apache License 2.0. See `LICENSE` and `NOTICE` for the original copyright and a list of changes.
