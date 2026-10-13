"""Opens a camera, a network stream or a video file and keeps frames coming.

A source can be a webcam number ("0"), a device path ("/dev/video0"), a stream URL
(rtsp://, http://, https://) or a video file. Files loop forever; live sources are reopened
with a growing delay when they cannot be reached or the connection drops.
"""
import time

LIVE_PREFIXES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "/dev/video")


def parse_source(text):
    """Return (source, live). A webcam number becomes an int, everything else stays a string."""
    text = str(text).strip()
    if text.isdigit():
        return int(text), True
    return text, text.lower().startswith(LIVE_PREFIXES)


def _open_with_opencv(source):
    import cv2  # imported here so this module can be tested without OpenCV
    return cv2.VideoCapture(source)


class FrameSource:
    """Reads frames from one source: loops video files and reconnects live cameras."""

    def __init__(self, text, configure=None, open_capture=_open_with_opencv, sleep=time.sleep,
                 max_delay=30.0, log=print):
        self.source, self.live = parse_source(text)
        self.reconnects = 0
        self._configure = configure
        self._open_capture = open_capture
        self._sleep = sleep
        self._max_delay = max_delay
        self._log = log
        self._cap = None

    def _open(self):
        self.release()
        cap = self._open_capture(self.source)
        if cap is None or not cap.isOpened():
            return False
        if self._configure:
            self._configure(cap)
        self._cap = cap
        return True

    def read(self, stop=lambda: False):
        """Return the next frame, or None if a file cannot be read or stop() becomes true."""
        delay = 1.0
        rewound = False
        while not stop():
            if self._cap is None and not self._open():
                if not self.live:
                    self._log("Could not open {}".format(self.source))
                    return None
                self._log("Camera {} not reachable, retrying in {:.0f}s".format(self.source, delay))
                self._sleep(delay)
                delay = min(delay * 2, self._max_delay)
                continue
            ok, frame = self._cap.read()
            if ok and frame is not None:
                return frame
            self.release()
            if self.live:
                self.reconnects += 1
                self._log("Lost camera {}, reconnecting in {:.0f}s".format(self.source, delay))
                self._sleep(delay)
                delay = min(delay * 2, self._max_delay)
            elif rewound:
                self._log("No frames in {}".format(self.source))
                return None
            else:
                rewound = True    # end of the video file: start it again
        return None

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
