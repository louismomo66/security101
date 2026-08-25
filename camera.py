"""OpenCV camera wrapper with background capture thread.

Mirrors the iOS CameraController + AsyncStream(.bufferingNewest(1)) pattern:
a daemon thread captures frames continuously; callers always get the most
recent frame, older ones are silently dropped.
"""

import threading
import time

import cv2


class OpenCVCamera:
    def __init__(self, device_id=0, width=640, height=480):
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            print("[OpenCVCamera] WARNING: could not open camera device", device_id)
            print("  → On macOS, grant camera access to your terminal app in")
            print("    System Settings → Privacy & Security → Camera.")
            self._running = False
            self._frame = None
            self._lock = threading.Lock()
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def is_open(self):
        return self._running

    def _loop(self):
        while self._running:
            ok, bgr = self.cap.read()
            if ok:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._frame = rgb
            else:
                time.sleep(0.01)

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def release(self):
        self._running = False
        if self.cap.isOpened():
            self.cap.release()
