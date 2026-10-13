import pytest

from camera import FrameSource, parse_source


@pytest.mark.parametrize("text, expected", [
    ("0", (0, True)),
    (" 2 ", (2, True)),
    ("/dev/video0", ("/dev/video0", True)),
    ("rtsp://user:pass@192.168.1.20:554/stream1", ("rtsp://user:pass@192.168.1.20:554/stream1", True)),
    ("http://cam.local/video.mjpg", ("http://cam.local/video.mjpg", True)),
    ("demo_video.mp4", ("demo_video.mp4", False)),
    ("C:/videos/site.avi", ("C:/videos/site.avi", False)),
])
def test_parse_source(text, expected):
    assert parse_source(text) == expected


class FakeCapture:
    """Plays a list of frames; None in the list means a failed read."""

    def __init__(self, frames, opened=True):
        self.frames = list(frames)
        self.opened = opened
        self.released = False
        self.configured = False

    def isOpened(self):
        return self.opened and not self.released

    def read(self):
        if not self.frames:
            return False, None
        frame = self.frames.pop(0)
        return frame is not None, frame

    def release(self):
        self.released = True


class Opener:
    """Hands out prepared captures in order and records what was opened."""

    def __init__(self, *captures):
        self.captures = list(captures)
        self.calls = []

    def __call__(self, source):
        self.calls.append(source)
        return self.captures.pop(0) if self.captures else FakeCapture([], opened=False)


def make(text, *captures, **kw):
    opener = Opener(*captures)
    sleeps = []
    src = FrameSource(text, open_capture=opener, sleep=sleeps.append, log=lambda *a: None, **kw)
    return src, opener, sleeps


def test_video_file_loops_back_to_the_start():
    src, opener, _ = make("clip.mp4", FakeCapture(["f1", "f2"]), FakeCapture(["f1", "f2"]))
    assert [src.read() for _ in range(4)] == ["f1", "f2", "f1", "f2"]
    assert opener.calls == ["clip.mp4", "clip.mp4"]


def test_missing_video_file_returns_none():
    src, _, _ = make("missing.mp4", FakeCapture([], opened=False))
    assert src.read() is None


def test_empty_video_file_returns_none_instead_of_spinning():
    src, _, _ = make("empty.mp4", FakeCapture([]), FakeCapture([]))
    assert src.read() is None


def test_webcam_number_is_opened_as_an_int():
    src, opener, _ = make("0", FakeCapture(["f1"]))
    assert src.read() == "f1"
    assert opener.calls == [0]


def test_live_camera_reconnects_after_a_dropped_stream():
    first = FakeCapture(["f1", None])
    src, opener, sleeps = make("rtsp://cam/stream", first, FakeCapture(["f2"]))
    assert src.read() == "f1"
    assert src.read() == "f2"
    assert first.released
    assert src.reconnects == 1
    assert sleeps == [1.0]
    assert len(opener.calls) == 2


def test_unreachable_camera_backs_off_until_it_comes_up():
    down = [FakeCapture([], opened=False) for _ in range(6)]
    src, _, sleeps = make("rtsp://cam/stream", *down, FakeCapture(["f1"]), max_delay=8.0)
    assert src.read() == "f1"
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]


def test_stop_ends_a_wait_for_an_unreachable_camera():
    src, _, sleeps = make("rtsp://cam/stream")
    calls = []

    def stop():
        calls.append(1)
        return len(calls) > 3

    assert src.read(stop=stop) is None
    assert len(sleeps) == 3


def test_configure_runs_on_every_open():
    caps = [FakeCapture(["f1", None]), FakeCapture(["f2"])]
    seen = []

    def configure(cap):
        seen.append(cap)

    src, _, _ = make("0", *caps, configure=configure)
    src.read()
    src.read()
    assert seen == caps
