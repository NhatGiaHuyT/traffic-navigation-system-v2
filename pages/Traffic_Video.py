import streamlit as st
import cv2
import torch
import numpy as np
import warnings
import base64
import time
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from datetime import datetime
import threading
import queue
import os
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import shutil
import re
torch.classes.__path__ = []

warnings.filterwarnings("ignore")

# Setup logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = r"../models/yolov8_d5_100e.pt"
SCALE_FACTOR = 0.05

class_names = [
    "Motorbike", "Car", "Bus", "Truck",
    "Motorbike(night)", "Car(night)", "Bus(night)", "Truck(night)"
]

ROI_POLYGON = np.array([
    [200, 200],
    [1100, 200],
    [1100, 600],
    [200, 600]
], dtype=np.int32)

def is_center_in_roi(x1, y1, x2, y2, polygon):
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    result = cv2.pointPolygonTest(polygon, (center_x, center_y), False)
    return result >= 0

def calculate_iou(boxA, boxB):
    boxA = [boxA[0], boxA[1], boxA[2] - boxA[0], boxA[3] - boxA[1]]
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB
    x1 = max(xA, xB)
    y1 = max(yA, yB)
    x2 = min(xA + wA, xB + wB)
    y2 = min(yA + hA, yB + hB)
    if x2 < x1 or y2 < y1:
        return 0.0
    interArea = (x2 - x1) * (y2 - y1)
    boxAArea = wA * hA
    boxBArea = wB * hB
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def detect_direction(track, x1, y1):
    if hasattr(track, 'prev_position_dir'):
        prev_x, prev_y = track.prev_position_dir
        dx = x1 - prev_x
        dy = y1 - prev_y
        direction = ""
        if abs(dx) > abs(dy):
            direction = "Right" if dx > 0 else "Left"
        else:
            direction = "Down" if dy > 0 else "Up"
        track.prev_position_dir = [x1, y1]
        return direction
    else:
        track.prev_position_dir = [x1, y1]
        return "Unknown"

def init_models(model_path):
    model = YOLO(model_path)
    tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
    return model, tracker

track_class_map = {}

def process_frame(frame, model, tracker, conf_threshold=0.5):
    global track_class_map
    track_class_map.clear()

    results = model(frame)
    if isinstance(results, list):
        results = results[0]
    detections = []

    if results.boxes is not None and results.boxes.data is not None:
        for det in results.boxes.data:
            x1, y1, x2, y2, conf, cls = det.tolist()
            if conf < conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            class_name = model.names[int(cls)]
            detections.append([[x1, y1, x2 - x1, y2 - y1], float(conf), int(cls), class_name])

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_bbox = track.to_ltrb()
        best_match = None
        best_iou = 0
        for detection in detections:
            detection_bbox = detection[0]
            iou = calculate_iou(track_bbox, detection_bbox)
            if iou > best_iou:
                best_iou = iou
                best_match = detection
        if best_match and best_iou > 0.3:
            track_class_map[track.track_id] = best_match[3]
        else:
            if hasattr(track, 'det_class_name'):
                track_class_map[track.track_id] = track.det_class_name
            else:
                track_class_map[track.track_id] = 'Unknown'
    return tracks

def draw_tracks(frame, tracks, scale_factor,
                color_by_speed=False,
                direction_detection=False):
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        class_name = track_class_map.get(track_id, 'Unknown')

        if hasattr(track, 'prev_position'):
            prev_position = track.prev_position
            distance_pixels = np.linalg.norm(np.array([x1, y1]) - np.array(prev_position))
            speed_mps = distance_pixels * scale_factor
            speed_text = f"Speed: {speed_mps:.2f} m/s"
            track.prev_position = [x1, y1]
        else:
            speed_text = "Speed: N/A"
            track.prev_position = [x1, y1]

        direction_text = ""
        if direction_detection:
            direction_text = detect_direction(track, x1, y1)

        box_color = (0, 255, 0)
        if color_by_speed and "Speed: N/A" not in speed_text:
            speed_val = float(speed_text.split(":")[1].replace("m/s", "").strip())
            if speed_val < 5:
                box_color = (0, 255, 0)
            elif 5 <= speed_val < 10:
                box_color = (0, 165, 255)
            else:
                box_color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        cv2.putText(frame, class_name, (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, speed_text, (x1, y1 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if direction_text:
            cv2.putText(frame, f"Dir: {direction_text}", (x1, y1 - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame

def get_vehicle_counts_in_roi(tracks, polygon):
    total_in_roi = 0
    class_counts = defaultdict(int)
    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        if is_center_in_roi(x1, y1, x2, y2, polygon):
            total_in_roi += 1
            cls_name = track_class_map.get(track.track_id, 'Unknown')
            class_counts[cls_name] += 1
    return total_in_roi, dict(class_counts)

def get_congestion_level(total_vehicle_count, threshold_medium, threshold_heavy):
    if total_vehicle_count >= threshold_heavy:
        return 'Heavy'
    elif total_vehicle_count >= threshold_medium:
        return 'Medium'
    else:
        return 'Light'

def display_congestion_notification(congestion_class, placeholder):
    colors = {
        'Heavy': ('rgba(255, 0, 0, 0.5)', 'red'),
        'Medium': ('rgba(255, 165, 0, 0.5)', 'orange'),
        'Light': ('rgba(0, 255, 0, 0.5)', 'green'),
    }
    bg_color, border_color = colors.get(congestion_class, ('rgba(0, 0, 0, 0.5)', 'black'))
    placeholder.markdown(
        f"""
        <div style="background-color: {bg_color}; border: 2px solid {border_color}; border-radius: 10px; padding: 10px; text-align: center;">
            <strong style="color: black; font-size: 20px;">{congestion_class} Congestion Detected!</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

def trigger_alert(congestion_class, alert_mode="none"):
    if congestion_class == "Heavy":
        if alert_mode == "audio":
            pass
        elif alert_mode == "webhook":
            pass

def log_data_to_csv(log_df, csv_file="../data/traffic_log.csv"):
    if not os.path.exists(csv_file):
        log_df.to_csv(csv_file, index=False)
    else:
        log_df.to_csv(csv_file, mode='a', header=False, index=False)

def write_live_traffic_data(total_in_roi, congestion_class, timestamp, csv_file="../data/traffic_live_data.csv"):
    data = {
        "timestamp": [timestamp],
        "vehicles_in_roi": [total_in_roi],
        "congestion_level": [congestion_class]
    }
    df = pd.DataFrame(data)
    temp_file = csv_file + ".tmp"
    df.to_csv(temp_file, index=False)
    shutil.move(temp_file, csv_file)

# Load origin-destination-route mapping CSV once
_mapping_df = pd.read_csv("../data/od_routes_mapping.csv")
_mapping_df.columns = [col.strip() for col in _mapping_df.columns]

import re

# Parse the Route string into list of routes for each row
def parse_route_str(route_str):
    # Use regex to find all occurrences of 'Đường ...' inside the string
    # The pattern looks for 'Đường' followed by any non-quote characters
    pattern = r"'([^']+)'"
    matches = re.findall(pattern, route_str)
    return matches

# Build a mapping from individual route to list of (Origin, Destination)
_ROUTE_TO_OD = {}
for _, row in _mapping_df.iterrows():
    origin = row['Origin']
    destination = row['Destination']
    route_str = row['Route']
    # Parse route string into individual routes
    route_list = parse_route_str(route_str)
    for route in route_list:
        if route not in _ROUTE_TO_OD:
            _ROUTE_TO_OD[route] = []
        _ROUTE_TO_OD[route].append((origin, destination))

_CONGESTION_INTENSITY = {
    "Light": 10,
    "Medium": 50,
    "Heavy": 90
}

def write_cnn_output_for_lstm(route, vehicle_count, congestion_level, timestamp, csv_file="../data/cnn_output_for_lstm.csv"):
    weather_condition = "Clear"
    accident_reports = 0
    traffic_intensity = _CONGESTION_INTENSITY.get(congestion_level, vehicle_count)
    try:
        dt = datetime.fromisoformat(timestamp)
    except ValueError:
        dt = datetime.now()
    date_str = dt.strftime("%Y-%m-%d")
    time_str = dt.strftime("%H:%M:%S")
    day_of_week = dt.strftime("%A")
    distance = 0

    # Determine if route is a single route or a concatenated string of routes
    if route.startswith("[") and route.endswith("]"):
        routes = parse_route_str(route)
    else:
        routes = [route]

    rows = []
    for r in routes:
        od_list = _ROUTE_TO_OD.get(r, [("Unknown", "Unknown")])
        for origin, destination in od_list:
            row = {
                "Date": date_str,
                "Time": time_str,
                "Day of the Week": day_of_week,
                "Origin": origin,
                "Destination": destination,
                "Route": r,
                "Weather Conditions": weather_condition,
                "Accident Reports": accident_reports,
                "Traffic Intensity": traffic_intensity,
                "Distance": distance
            }
            rows.append(row)

    df_new = pd.DataFrame(rows)
    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
        try:
            df_existing = pd.read_csv(csv_file)
            # Check if 'Date' and 'Time' columns exist for filtering
            if 'Date' in df_existing.columns and 'Time' in df_existing.columns:
                mask = ~(
                    (df_existing['Route'].isin(routes)) &
                    (df_existing['Date'] == date_str) &
                    (df_existing['Time'] == time_str)
                )
            elif 'Timestamp' in df_existing.columns:
                # fallback to old filtering by Timestamp and Route
                iso_ts = dt.isoformat()
                mask = ~((df_existing['Route'].isin(routes)) & (df_existing['Timestamp'] == iso_ts))
            else:
                # no filtering if columns missing
                mask = pd.Series([True] * len(df_existing))
            df_combined = pd.concat([df_existing[mask], df_new], ignore_index=True)
        except Exception as e:
            logger.error(f"Error reading or processing existing CSV file: {e}")
            df_combined = df_new
    else:
        df_combined = df_new
    temp_file = f"{csv_file}.tmp"
    df_combined.to_csv(temp_file, index=False)
    shutil.move(temp_file, csv_file)

#VIDEO_SOURCES = {
#    "Đường Bạch Đằng": r"Traffic2.mp4",
#    "Đường Nguyễn Văn Linh": r"Traffic2.mp4",
#    "Đường Lê Văn Sỹ": r"Traffic2.mp4",
#    "Đường Đồng Khởi": r"Traffic2.mp4",
#    "Đường Trường Chinh": r"Traffic2.mp4",
#    "Đường Nguyễn Văn Quá": r"Traffic2.mp4",
#    "Đường Lý Thường Kiệt": r"Traffic2.mp4",
#    "Đường Cái Vạn": r"Traffic2.mp4",
#    "Đường Tô Hiệu": r"Traffic2.mp4",
#    "Đường Pasteur": r"Traffic2.mp4",
#    "Đường Lý Tự Trọng": r"Traffic2.mp4",
#    "Đường Cách Mạng Tháng Tám": r"Traffic2.mp4",
#    "Đường Lê Lợi": r"Traffic2.mp4",
#    "Đường 3/2": r"Traffic2.mp4",
#    "Đường Trần Hưng Đạo": r"Traffic2.mp4",
#}

VIDEO_SOURCES = {
    route: r"../videos/Traffic2.mp4"
    for route in _ROUTE_TO_OD.keys()
}

data_lock = threading.Lock()
latest_data = {}

def camera_worker(cam_id, src, model, tracker, threshold_medium, threshold_heavy):
    cap = cv2.VideoCapture(src)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 360))
        results = model(frame)[0]
        detections = []
        if results.boxes is not None and results.boxes.data is not None:
            for det in results.boxes.data:
                x1, y1, x2, y2, conf, cls = det.tolist()
                if conf < 0.5:
                    continue
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                class_name = model.names[int(cls)]
                detections.append([[x1, y1, x2 - x1, y2 - y1], float(conf), int(cls), class_name])
        tracks = tracker.update_tracks(detections, frame=frame)
        frame = draw_tracks(frame, tracks, scale_factor=SCALE_FACTOR, color_by_speed=True, direction_detection=True)
        cv2.polylines(frame, [ROI_POLYGON], True, (255, 255, 0), 2)
        total_in_roi, class_counts_roi = get_vehicle_counts_in_roi(tracks, ROI_POLYGON)
        congestion_class = get_congestion_level(total_in_roi, threshold_medium, threshold_heavy)
        timestamp = datetime.now().isoformat()
        write_cnn_output_for_lstm(cam_id, total_in_roi, congestion_class, timestamp)
        with data_lock:
            latest_data[cam_id] = {
                "frame": frame[:, :, ::-1],
                "total": total_in_roi,
                "congestion": congestion_class,
                "timestamp": timestamp,
                "vehicle_classes": class_counts_roi,
            }
        time.sleep(0.1)
    cap.release()

def main():
    st.markdown(
        """
        <h1 style='text-align: center; color: black; white-space: nowrap;'>
        <span style='color: #FF5733;'>Traffic Monitoring</span> Application
        </h1>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.header("Configuration & Controls")
    model_options = {
        "YOLOv8 (Default)": r"../models/yolov8_d5_100e.pt",
        "Custom YOLOv8 50 Epochs": r"./src/ckpts/yolov8_d5_50e.pt",
        "Custom YOLOv8 100 Epochs": r"./src/ckpts/yolov8_d5_100e.pt",
        "Custom YOLOv12 50 Epochs": r"./src/ckpts/yolov12_d5_50e.pt",
        "Custom YOLOv12 100 Epochs": r"./src/ckpts/yolov12_d5_100e.pt",
    }
    selected_model = st.sidebar.selectbox("YOLO Model Path", list(model_options.keys()), index=0)
    model_path = model_options[selected_model]
    threshold_medium = st.sidebar.slider("Medium Congestion Threshold", 1, 50, 8)
    threshold_heavy = st.sidebar.slider("Heavy Congestion Threshold", 5, 100, 15)
    if st.sidebar.button("Start Monitoring"):
        model, tracker = init_models(model_path)
        for cam_id, src in VIDEO_SOURCES.items():
            threading.Thread(
                target=camera_worker,
                args=(cam_id, src, model, tracker, threshold_medium, threshold_heavy),
                daemon=True,
            ).start()
    # Add camera selection for user to view one camera at a time
    cam_ids = list(VIDEO_SOURCES.keys())
    selected_cam = st.sidebar.selectbox("Select Camera to View", cam_ids)

    placeholders = {cam_id: st.empty() for cam_id in VIDEO_SOURCES}

    while True:
        with data_lock:
            snapshot = latest_data.copy()
        # Show only the selected camera's feed
        placeholder = placeholders[selected_cam]
        data = snapshot.get(selected_cam)
        with placeholder.container():
            st.subheader(selected_cam)
            if data:
                st.image(data["frame"], use_container_width=True)
                st.markdown(f"Total Vehicles: {data['total']}")
                st.markdown(f"Congestion Level: {data['congestion']}")
                st.markdown(f"Last Updated: {data['timestamp']}")
            else:
                st.write("Waiting for data...")
        time.sleep(1)

if __name__ == "__main__":
    main()
