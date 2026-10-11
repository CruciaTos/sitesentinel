#!/bin/sh
# Downloads the model and demo video, which are unchanged from the original
# CiscoDevNet/ppe-detection project (Apache License 2.0).
set -e
BASE=https://github.com/CiscoDevNet/ppe-detection/raw/master/inference
cd "$(dirname "$0")/inference"
mkdir -p model
curl -L --fail -o model/frozen_inference_graph.pb "$BASE/model/frozen_inference_graph.pb"
curl -L --fail -o demo_video.mp4 "$BASE/demo_video.mp4"
echo "Done. Model and demo video are in inference/."
