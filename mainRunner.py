import cv2
import easyocr
import threading
import queue
import time
import socket
import csv
import os
import requests
from datetime import datetime
from ultralytics import YOLO

# =========================
# HELPER: GET LOCAL IP
# =========================
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


# =========================
# helper: crop frame to center
# =========================
def center_crop(frame, crop_ratio=0.5):
    """
    Keep the full height and crop the width around the center.

    crop_ratio=0.5 keeps the central 50% of the width
    and 100% of the height.
    """
    height, width = frame.shape[:2]

    crop_width = int(width * crop_ratio)

    x1 = (width - crop_width) // 2
    x2 = x1 + crop_width

    return frame[:, x1:x2]

# =========================
# RTMP STREAM WITH AUTO-RECONNECT
# =========================
class ThreadedRTMPStream:
    """Continuously reads RTMP frames and reconnects after failures."""

    def __init__(self, stream_url, retry_delay=3, max_read_failures=30):
        self.stream_url = stream_url
        self.retry_delay = retry_delay
        self.max_read_failures = max_read_failures

        # Configure FFmpeg for low-latency RTMP.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "fflags;nobuffer|flags;low_delay|analyzeduration;0|probesize;32"
        )

        self.cap = None
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _connect(self):
        """Keep retrying until the RTMP stream becomes available."""
        while self.running:
            print(f"[RTMP] Connecting to {self.stream_url} ...")

            cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)

            if cap.isOpened():
                with self.lock:
                    self.cap = cap
                    self.ret = False
                    self.frame = None

                print("[RTMP] Connected successfully.")
                return True

            cap.release()
            print(f"[RTMP] Connection failed. Retrying in {self.retry_delay}s...")
            time.sleep(self.retry_delay)

        return False

    def _update(self):
        failed_reads = 0

        while self.running:
            with self.lock:
                cap = self.cap

            if cap is None:
                if not self._connect():
                    break
                failed_reads = 0
                continue

            ret, frame = cap.read()

            if ret and frame is not None:
                with self.lock:
                    self.ret = True
                    self.frame = frame
                failed_reads = 0
            else:
                failed_reads += 1

                if failed_reads >= self.max_read_failures:
                    print("[RTMP] Stream lost. Reconnecting...")

                    with self.lock:
                        old_cap = self.cap
                        self.cap = None
                        self.ret = False
                        self.frame = None

                    if old_cap is not None:
                        old_cap.release()

                    failed_reads = 0
                    time.sleep(self.retry_delay)
                else:
                    time.sleep(0.05)

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def is_opened(self):
        with self.lock:
            return self.cap is not None and self.cap.isOpened()

    def release(self):
        self.running = False

        with self.lock:
            cap = self.cap
            self.cap = None
            self.ret = False
            self.frame = None

        if cap is not None:
            cap.release()

        if self.thread.is_alive():
            self.thread.join(timeout=2)

        print("[RTMP] Stream reader stopped.")


# =========================
# INITIALIZATION
# =========================
LOCAL_IP = get_local_ip()
RTMP_URL = "rtmp://localhost:1935/live/drone"

LOCAL_API_URL = "http://127.0.0.1:8000/api/annotate-latest"
API_KEY = "forest"
DEVICE_ID = "TDM2421-01"

print("=" * 60)
print(f"1.  make sure mediamtx is already running")
print(f"2.  in DJI Fly, set RTMP URL to:")
print(f"        rtmp://{LOCAL_IP}:1935/live/drone")
print(f"3.  run this script")
print(f"4.  W forest gaming")
print("=" * 60)

# Load YOLO Model
model = YOLO(r"./runs/classify/train/weights/best.pt")

# CSV Logging Initialization
CSV_FILE = "forest_log.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "forest_class", "confidence", "telemetry"])

# =========================
# MAIN STREAM LOOP
# =========================
print(f"Connecting to local RTMP stream: {RTMP_URL} ...")
stream = ThreadedRTMPStream(RTMP_URL)

frame_count = 0
forest_class = "waiting..."
conf = 0.0
send_count = 1

while True:
    ret, frame = stream.read()

    if not ret or frame is None:
        # Stream has not started yet or is re-buffering
        time.sleep(0.05)
        continue

    frame_count += 1

    # YOLO Inference (Every 10 frames)
    if frame_count % 10 == 0:
        try:
            # set displayed forest type
            inferFrame = center_crop(frame)
            small_frame = cv2.resize(inferFrame, (384, 384))
            results = model(small_frame, verbose=False)
            probs = results[0].probs
            class_id = probs.top1
            conf = float(probs.top1conf)
            forest_class = model.names[class_id]

            # Send the latest forest classification to the local API.
            try:
                # Normalize model labels to the categories accepted by server.py.
                label = str(forest_class).strip().lower()

                aliases = {
                    "dense": "dense",
                    "medium": "medium",
                    "sparse": "sparse",
                    "unknown": "unknown",
                }

                density = aliases.get(label, "unknown")

                response = requests.post(
                    LOCAL_API_URL,
                    params={
                        "density": density,
                        "confidence": conf,
                    },
                    headers={"X-API-Key": API_KEY},
                    timeout=3,
                )

                print(f"uploaded #{send_count}")
                send_count += 1

                # if response.status_code not in (200, 201):
                #     print(f"Classification upload failed ({response.status_code}), got response: ", response.text)

            except requests.RequestException as e:
                print("Classification API error:", e)
        except Exception as e:
            print("YOLO Inference Error:", e)

    # Log to CSV (Every 60 frames)
    if frame_count % 30 == 0:
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), forest_class, round(conf, 3)])

    # Overlay HUD
    cv2.rectangle(frame, (10, 10), (900, 120), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Forest Density: {forest_class} ({conf:.2f})",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    # show the parts thats being cropped for inference
    display_frame = frame.copy()
    
    height, width = frame.shape[:2]
    crop_width = int(width * 0.5)

    x1 = (width - crop_width) // 2
    x2 = x1 + crop_width

    cv2.rectangle(
        display_frame,
        (x1, 0),
        (x2, height - 1),
        (0, 255, 0),
        2
    )

    cv2.imshow("Live Drone RTMP Feed", display_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

# Cleanup
stream.release()
cv2.destroyAllWindows()