# -*- coding: utf-8 -*-
"""Runs the PPE detector on a camera, stream or video file and reports each check to the backend.

Examples:
    python3 run_detector.py --model_dir=model --source=demo_video.mp4 --camera_id=gate-1 --show_video_window=1
    python3 run_detector.py --model_dir=model --source=0 --camera_id=office-webcam
    python3 run_detector.py --model_dir=model --source=rtsp://user:pass@192.168.1.20:554/stream1 --camera_id=gate-1

Run one detector per camera. Live sources reconnect by themselves when the connection drops.

Derived from CiscoDevNet/ppe-detection (Apache License 2.0); see NOTICE.
"""
import argparse
import base64
import os
import queue
import time
from multiprocessing import Process, Queue, Value

import cv2
import numpy as np
import requests
import tensorflow as _tf

import compliance
import config
import messages
import overlay
from camera import FrameSource

# The model is a TensorFlow 1.x frozen graph; TF 2.x runs it through the v1 API.
tf = getattr(_tf, 'compat', None) and _tf.compat.v1 or _tf
_version = tuple(int(x) for x in _tf.__version__.split('.')[:2])
if _version < (1, 12):
    raise ImportError('SiteSentinel needs TensorFlow 1.12 or newer.')

WINDOW = 'SiteSentinel'
CATEGORY_NAMES = {1: 'hardhat', 2: 'vest', 3: 'person'}
OUTPUT_KEYS = ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']


def load_model(path):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(path, 'rb') as fid:
            graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def, name='')
    return graph


def output_tensors():
    """Look up the model's output tensors in the default graph."""
    names = {out.name for op in tf.get_default_graph().get_operations() for out in op.outputs}
    return {key: tf.get_default_graph().get_tensor_by_name(key + ':0')
            for key in OUTPUT_KEYS if key + ':0' in names}


def detect(frame, sess, tensors):
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    # OpenCV frames are BGR but the model was trained on RGB, so convert before detecting.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out = sess.run(tensors, feed_dict={image_tensor: np.expand_dims(rgb, axis=0)})
    out['num_detections'] = int(out['num_detections'][0])
    out['detection_classes'] = out['detection_classes'][0].astype(np.int64)
    out['detection_boxes'] = out['detection_boxes'][0]
    out['detection_scores'] = out['detection_scores'][0]
    return out


def encode_snapshot(image):
    ok, jpeg = cv2.imencode('.jpg', image)
    return {
        'height': image.shape[0],
        'width': image.shape[1],
        'format': 'jpeg',
        'raw': base64.b64encode(jpeg.tobytes()).decode('utf-8'),
    }


def report_worker(run_flag, outbox):
    """Separate process: posts queued (url, message) pairs so inference never waits on the network."""
    while run_flag.value:
        try:
            url, message = outbox.get(block=True, timeout=5)
        except queue.Empty:
            continue
        try:
            response = requests.post(url, json=message, timeout=5)
            if response.status_code >= 400:
                print('Backend rejected the check ({}): {}'.format(response.status_code, response.text[:200]))
        except requests.exceptions.RequestException:
            print('Could not reach the backend at', url)


def open_window():
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    mode = cv2.WINDOW_FULLSCREEN if config.display_full_screen else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, mode)


def set_resolution(cap):
    if (config.capture_image_width, config.capture_image_height) in config.supported_video_resolution:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.capture_image_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.capture_image_height)


def process_video(graph, source_text, show_window, camera_id, report, run_flag, outbox):
    source = FrameSource(source_text, configure=set_resolution)
    print('Reading', 'live camera' if source.live else 'video file', source.source, 'as', camera_id)
    if show_window:
        open_window()

    threshold = config.object_confidence_threshold
    with graph.as_default():
        tensors = output_tensors()
        with tf.Session() as sess:
            last_sent = time.time()
            while run_flag.value:
                frame = source.read(stop=lambda: not run_flag.value)
                if frame is None:
                    break

                output = detect(frame, sess, tensors)
                # The overlay always uses the quick overlap rule; with --report frames the backend's
                # trained model makes the decision that is recorded.
                people = compliance.assess_people(output, threshold)
                overlay.draw_frame(frame, output, people, camera_id, threshold)

                if people and time.time() - last_sent > config.message_send_interval / 1000.0:
                    snapshot = cv2.resize(frame, (config.storage_image_width, config.storage_image_height))
                    image = encode_snapshot(snapshot)
                    if report == 'frames':
                        item = (config.frames_api, messages.frame_message(camera_id, output, threshold, image))
                    else:
                        item = (config.detection_api, messages.detection_message(camera_id, people, image))
                    try:
                        outbox.put_nowait(item)
                        last_sent = time.time()
                    except queue.Full:
                        pass

                if show_window:
                    cv2.imshow(WINDOW, cv2.resize(frame, (config.display_window_width, config.display_window_height)))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        run_flag.value = 0
                        break

    source.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Hard hat and vest detection')
    parser.add_argument('--model_dir', required=True, help='folder containing frozen_inference_graph.pb')
    parser.add_argument('--source', '--video_file_name', dest='source', required=True,
                        help='webcam number (0), device (/dev/video0), stream URL (rtsp://...) or video file')
    parser.add_argument('--camera_id', required=True, help='a name for this camera, shown on the dashboard')
    parser.add_argument('--show_video_window', type=int, default=0, help='1 to show the preview window, 0 to hide it')
    parser.add_argument('--report', choices=['frames', 'detections'], default='frames',
                        help='frames: send boxes and let the backend model decide (default); '
                             'detections: send the overlap rule decisions')
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, 'frozen_inference_graph.pb')
    if not os.path.exists(model_path):
        print('Could not find frozen_inference_graph.pb in', args.model_dir)
        raise SystemExit(1)

    print('Loading model')
    graph = load_model(model_path)

    run_flag = Value('i', 1)
    outbox = Queue(1)
    worker = Process(target=report_worker, args=(run_flag, outbox))
    worker.start()
    try:
        process_video(graph, args.source, bool(args.show_video_window),
                      args.camera_id, args.report, run_flag, outbox)
    finally:
        run_flag.value = 0
        worker.join()


if __name__ == '__main__':
    main()
