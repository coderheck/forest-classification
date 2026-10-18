# ===== script dùng scrcpy =====

import cv2
import numpy as np
import mss
import pygetwindow as gw
import time
import subprocess
import easyocr
import csv
from datetime import datetime
import os
import serial
import threading

# ============================================================
# SCRCPY
# ============================================================

SCRCPY_TITLE = "camcamcam_forest-clc"
scrcpy_process = subprocess.Popen([
    "./scrcpy/scrcpy.exe",
    "-d",  # auto-select connected via usb
    # "--tcpip=+192.168.137.41",  # specify ip (last number always different on each connection)
    # "--tcpip", # config device to use proper tcpip settings for adb
    "--video-codec=h264",
    "--max-size=1800",
    "--max-fps=60",
    "--no-audio",
    "--window-borderless",
    "--always-on-top",
    f"--window-title={SCRCPY_TITLE}",
    # "--window-height=1400",
])

# ============================================================
# FIND SCRCPY WINDOW
# ============================================================

# wait until scrcpy screen is available
wait_tt = 0
while not gw.getWindowsWithTitle(SCRCPY_TITLE):
    time.sleep(1)
    wait_tt += 1
    print(f"waiting for scrcpy window... ({wait_tt} secs)")

try:
    windows = gw.getWindowsWithTitle(SCRCPY_TITLE)
    if not windows:
        raise IndexError

    win = windows[0]
    print("Found scrcpy window:")
    print(f"Position: {win.left}, {win.top}")
    print(f"Size: {win.width} x {win.height}")
    # win.activate()

except IndexError:
    print("Could not find scrcpy window.")
    scrcpy_process.terminate()
    exit()

# ============================================================
# OCR
# ============================================================

reader = easyocr.Reader(["en"], gpu=False, verbose=False)

# ============================================================
# CSV
# ============================================================

CSV_FILE = "forest_log.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["time", "lat", "lon", "forest_class", "confidence", "telemetry"]
        )

# ============================================================
# MODEL
# ============================================================

from ultralytics import YOLO

model = YOLO(r"./runs/classify/train/weights/best.pt", verbose=False)

# ============================================================
# VARIABLES
# ============================================================

frame_count = 0
forest_class = "unknown"
conf = 0.0
telemetry = "NO DATA"

# Fake GPS for now
lat = 10.7623
lon = 106.6821


# ============================================================
# REAL GPS
# ============================================================

GPS_PORT = "COM12"  # Change this to your board's COM port
GPS_BAUD = 115200

lat = None
lon = None
gps_status = "NO GPS FIX"


def nmea_to_decimal(value, direction):
    """Convert ddmm.mmmm / dddmm.mmmm to decimal degrees."""

    if not value or not direction:
        return None

    try:
        value = float(value)
        degrees = int(value // 100)
        minutes = value - degrees * 100
        decimal = degrees + minutes / 60

        if direction in ("S", "W"):
            decimal = -decimal

        return decimal

    except (ValueError, TypeError):
        return None


def read_gps():
    global lat, lon, gps_status

    try:
        gps = serial.Serial(
            GPS_PORT,
            GPS_BAUD,
            timeout=1
        )

        gps_status = "GPS CONNECTED"

        while True:
            line = gps.readline().decode(
                "ascii",
                errors="ignore"
            ).strip()

            if not line.startswith("LOC,"):
                continue

            # Example:
            # LOC,091644.00,1051.1155N,10647.0237E,...

            fields = line.split(",")

            if len(fields) < 4:
                continue

            latitude = fields[2].strip()
            longitude = fields[3].strip()

            if not latitude or not longitude:
                continue

            lat_dir = latitude[-1]
            lon_dir = longitude[-1]

            new_lat = nmea_to_decimal(
                latitude[:-1],
                lat_dir
            )

            new_lon = nmea_to_decimal(
                longitude[:-1],
                lon_dir
            )

            if new_lat is not None and new_lon is not None:
                lat = new_lat
                lon = new_lon
                gps_status = "GPS FIX"

    except Exception as e:
        gps_status = f"GPS ERROR: {e}"
        print(gps_status)


threading.Thread(
    target=read_gps,
    daemon=True
).start()


# ============================================================
# NORMALIZED OCR CROP
# ============================================================
#
# Instead of:
#
#     frame[430:470, 250:600]
#
# use percentages of the frame dimensions.
#
# These values assume the DJI telemetry is roughly in the
# lower portion of the screen.
#
# Adjust these four values if necessary.
#
# x1 = 0.20  -> 20% from left
# x2 = 0.80  -> 80% from left
# y1 = 0.85  -> 85% from top
# y2 = 0.98  -> 98% from top
#
# ============================================================

# OCR_X1 = 0.20
# OCR_X2 = 0.80
# OCR_Y1 = 0.85
# OCR_Y2 = 0.98
OCR_X1 = 1
OCR_X2 = 1
OCR_Y1 = 1
OCR_Y2 = 1


def get_ocr_crop(frame):
    height, width = frame.shape[:2]
    x1 = int(width * OCR_X1)
    x2 = int(width * OCR_X2)
    y1 = int(height * OCR_Y1)
    y2 = int(height * OCR_Y2)
    return frame[y1:y2, x1:x2]


# ============================================================
# HEATMAP SETTINGS
# ============================================================

HEATMAP_ROWS = 5
HEATMAP_COLS = 5

# How often the heatmap is recalculated.
#
# At 60 FPS:
#
# 10 frames ≈ 0.17 seconds
#
# Increase this to 20 or 30 if inference is too slow.

HEATMAP_INTERVAL = 10

# Amount of overlap between neighboring crops.
#
# 0.25 = 25% overlap
#
# Higher overlap gives a smoother heatmap but requires
# more computation if you increase the number of tiles.

HEATMAP_OVERLAP = 0.0

# Heatmap opacity.
#
# 0.0 = invisible
# 1.0 = completely covers camera feed

HEATMAP_OPACITY = 0.30

# ============================================================
# HEATMAP COLOR
# ============================================================


def density_to_color(class_name):
    """
    Convert forest density class into BGR color.

    dense  -> red
    medium -> yellow
    sparse -> green
    """

    if class_name == "dense":
        return (0, 0, 255)

    if class_name == "medium":
        return (0, 255, 255)

    if class_name == "sparse":
        return (0, 255, 0)

    return (255, 255, 255)


# ============================================================
# GENERATE HEATMAP CROPS
# ============================================================

INFERENCE_WIDTH_RATIO = 0.5

def get_heatmap_crops(frame):

    height, width = frame.shape[:2]

    crop_width = int(width * INFERENCE_WIDTH_RATIO)
    x1 = (width - crop_width) // 2
    x2 = x1 + crop_width

    crop = frame[:, x1:x2]

    return [crop], [(x1, 0, x2, height)]


# ============================================================
# CENTER-CROP INFERENCE
# ============================================================

def predict_heatmap(frame):
    crops, positions = get_heatmap_crops(frame)

    try:
        # Match mainRunner.py's classification input size.
        inference_images = [
            cv2.resize(crop, (384, 384))
            for crop in crops
        ]

        results = model(inference_images, verbose=False)

    except Exception as e:
        print("CENTER-CROP YOLO ERROR:", e)
        return None

    predictions = []

    for result, position in zip(results, positions):
        probs = result.probs
        class_id = probs.top1

        predictions.append({
            "class": str(model.names[class_id]).strip().lower(),
            "confidence": float(probs.top1conf),
            "position": position,
        })

    return predictions


# ============================================================
# DRAW HEATMAP
# ============================================================


def draw_heatmap(frame, predictions):

    if predictions is None:
        return frame

    # Separate image for the colored overlay.
    overlay = frame.copy()

    for prediction in predictions:

        class_name = prediction["class"]
        confidence = prediction["confidence"]

        x1, y1, x2, y2 = prediction["position"]

        color = density_to_color(class_name)

        # ----------------------------------------------------
        # Confidence controls how strongly the tile is shown.
        #
        # 0.0 confidence -> very weak
        # 1.0 confidence -> strong
        # ----------------------------------------------------

        alpha = 0.10 + (0.50 * confidence)

        alpha = min(alpha, 0.60)

        # Draw colored tile.
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    # --------------------------------------------------------
    # Blend the entire heatmap with the camera frame.
    # --------------------------------------------------------

    result = cv2.addWeighted(overlay, HEATMAP_OPACITY, frame, 1.0 - HEATMAP_OPACITY, 0)

    # --------------------------------------------------------
    # Draw boundaries and labels afterward so that they
    # remain readable.
    # --------------------------------------------------------

    for prediction in predictions:

        class_name = prediction["class"]
        confidence = prediction["confidence"]

        x1, y1, x2, y2 = prediction["position"]

        color = density_to_color(class_name)

        # Tile border

        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # Label

        label = f"{class_name} " f"{confidence:.2f}"

        # Background rectangle for text

        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        text_x = x1 + 5
        text_y = y1 + 25

        cv2.rectangle(
            result,
            (text_x - 2, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 2, text_y + 5),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            result, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    return result


# ============================================================
# SCREEN CAPTURE
# ============================================================

heatmap_predictions = None

with mss.MSS() as sct:
    while True:
        # ----------------------------------------------------
        # Get current scrcpy window position
        # ----------------------------------------------------

        monitor = {
            "top": win.top,
            "left": win.left,
            "width": win.width,
            "height": win.height,
        }

        # ----------------------------------------------------
        # Capture scrcpy
        # ----------------------------------------------------

        frame = np.array(sct.grab(monitor))
        processed = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # ----------------------------------------------------
        # Frame counter
        # ----------------------------------------------------

        frame_count += 1

        # ====================================================
        # CENTER-CROP INFERENCE
        # ====================================================

        if frame_count % HEATMAP_INTERVAL == 0:
            heatmap_predictions = predict_heatmap(processed)

        # ====================================================
        # DRAW HEATMAP
        # ====================================================

        display_frame = processed.copy()

        if heatmap_predictions is not None:
            display_frame = draw_heatmap(
                display_frame,
                heatmap_predictions
            )

        # Current forest classification comes from the center crop.
        if heatmap_predictions:
            forest_class = heatmap_predictions[0]["class"]
            conf = heatmap_predictions[0]["confidence"]

        # ====================================================
        # YOLO
        # ====================================================
        #
        # Don't run YOLO on every frame.
        # The forest isn't changing at 60 FPS.
        # Run it every 10 frames instead.
        #
        # ====================================================

        # if frame_count % 10 == 0:
        #     try:
        #         results = model(
        #             processed,
        #             # frame,
        #             verbose=False,
        #         )
        #         probs = results[0].probs
        #         class_id = probs.top1
        #         conf = float(probs.top1conf)
        #         forest_class = model.names[class_id]
        #     except Exception as e:
        #         print("YOLO ERROR:", e)

        # ====================================================
        # OCR
        # ====================================================

        if frame_count % 30 == 0:
            try:
                crop = get_ocr_crop(processed)
                if crop.size != 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    # Increase contrast for OCR
                    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

                    texts = reader.readtext(thresh, detail=0)

                    if texts:
                        telemetry = " ".join(texts)
            except Exception as e:
                print("OCR ERROR:", e)

        # ====================================================
        # SAVE CSV
        # ====================================================

        if frame_count % 60 == 0:
            with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"), 
                    lat, lon, 
                    forest_class, 
                    round(conf, 3), 
                    telemetry
                ])

        # ====================================================
        # DRAW OCR CROP RECTANGLE
        # ====================================================
        # This is extremely useful while configuring the
        # telemetry crop.
        # ====================================================

        height, width = frame.shape[:2]
        x1 = int(width * OCR_X1)
        x2 = int(width * OCR_X2)
        y1 = int(height * OCR_Y1)
        y2 = int(height * OCR_Y2)

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

        # ====================================================
        # HUD
        # ====================================================

        cv2.rectangle(frame, (10, 10), (1000, 120), (0, 0, 0), -1)

        cv2.putText(
            frame,
            f"Forest: {forest_class} ({conf:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Telemetry: {telemetry}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            display_frame,
            f"GPS: {gps_status}",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow("Drone Forest AI", display_frame)

        # ESC
        if cv2.waitKey(1) == 27:
            break


# ============================================================
# CLEANUP
# ============================================================

cv2.destroyAllWindows()

scrcpy_process.terminate()
