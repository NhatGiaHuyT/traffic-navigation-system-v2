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
import yaml

torch.classes.__path__ = []

warnings.filterwarnings("ignore")

# Setup logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = r"./models/yolov8_d5_100e.pt"
SCALE_FACTOR = 0.05

# Load class names from data.yml
with open("data/data.yml", "r") as f:
    data_config = yaml.safe_load(f)
class_names = data_config.get("names", [])

import streamlit as st
import matplotlib.colors as mcolors

ROI_POLYGON = np.array([
    [200, 200],
    [1100, 200],
    [1100, 600],
    [200, 600]
], dtype=np.int32)

def generate_class_colors(class_names):
    np.random.seed(42)
    colors = {}
    for idx in range(len(class_names)):
        colors[idx] = tuple(int(c) for c in np.random.randint(0, 256, size=3))
    return colors

def color_for_track(track_id):
    rng = np.random.RandomState(track_id)
    return tuple(int(c) for c in rng.randint(64, 256, size=3))


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
    # Removed setting model.names due to property setter error
    tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
    return model, tracker

track_class_map = {}

def process_frame(frame, model, tracker, conf_threshold=0.8):
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
            class_name = class_names[int(cls)]
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
        if best_match and best_iou > 0.2:
            track.det_class_name = best_match[3]
            track_class_map[track.track_id] = best_match[3]
        else:
            if hasattr(track, 'det_class_name'):
                track_class_map[track.track_id] = track.det_class_name
            else:
                track_class_map[track.track_id] = 'Unknown'
    return tracks

PIXEL_TO_METER = SCALE_FACTOR  # you can calibrate SCALE_FACTOR to meters per pixel

def draw_tracks(frame, tracks, track_states, class_colors,
                         use_class_color=True, use_track_color=False,
                         show_speed=True, show_direction=True,
                         custom_colors=None):
    """
    frame: BGR
    tracks: list of track objects
    track_states: dict mapping track_id -> state dict
    class_colors: dict mapping class_id -> (B,G,R)
    custom_colors: dict mapping class_id -> (B,G,R) or None
    """
    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        l, t, r, b = track.to_ltrb()
        x1, y1, x2, y2 = map(int, (l, t, r, b))
        cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)

        state = track_states.get(tid, {})
        cls_id = state.get('class_id', None)
        cls_name = state.get('class_name', 'Unknown')

        # Choose color
        if use_track_color:
            box_color = color_for_track(tid)
        elif custom_colors and cls_id is not None and cls_id in custom_colors:
            box_color = custom_colors[cls_id]
        elif use_class_color and cls_id is not None:
            box_color = class_colors.get(cls_id, (0, 255, 0))
        else:
            box_color = (0, 255, 0)

        # Compute speed and direction before updating prev_center/time
        curr_time = time.time()
        speed_text = ""
        if 'prev_center' in state and 'prev_time' in state:
            prev_cx, prev_cy = state['prev_center']
            prev_t = state['prev_time']
            dt = curr_time - prev_t
            if dt > 0:
                dist_px = np.linalg.norm(np.array([cx, cy]) - np.array([prev_cx, prev_cy]))
                speed_px_s = dist_px / dt
                speed_m_s = speed_px_s * PIXEL_TO_METER
                state['speed_m_s'] = speed_m_s
                if show_speed:
                    speed_text = f"{speed_m_s:.1f} m/s"
            # Direction
            if show_direction:
                dx = cx - prev_cx
                dy = cy - prev_cy
                # threshold to ignore jitter
                thresh = 3
                if abs(dx) < thresh and abs(dy) < thresh:
                    direction = "Static"
                elif abs(dx) > abs(dy):
                    direction = "Right" if dx > 0 else "Left"
                else:
                    direction = "Down" if dy > 0 else "Up"
                state['direction'] = direction
                direction_text = direction
            else:
                direction_text = ""
        else:
            # First time: initialize
            state['speed_m_s'] = 0.0
            if show_direction:
                state['direction'] = "Unknown"
                direction_text = "Unknown"
            speed_text = "0.0 m/s" if show_speed else ""
        # Update prev_center/time for next frame
        state['prev_center'] = (cx, cy)
        state['prev_time'] = curr_time
        track_states[tid] = state

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness=2)

        # Build label lines
        label_lines = [f"ID:{tid}", cls_name]
        if show_speed and speed_text:
            label_lines.append(speed_text)
        if show_direction and state.get('direction'):
            label_lines.append(state.get('direction'))
        # Draw labels stacked above box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thick = 1
        # Compute total height
        text_sizes = [cv2.getTextSize(line, font, font_scale, font_thick)[0] for line in label_lines]
        total_h = sum(h for (w,h) in text_sizes) + 4*len(label_lines)
        y0 = y1
        for i, line in enumerate(label_lines):
            (w, h), _ = cv2.getTextSize(line, font, font_scale, font_thick)
            # background rect: from (x1, y0 - h - 4) to (x1 + w + 2, y0)
            y_top = y0 - h - 4
            y_bottom = y0
            cv2.rectangle(frame, (x1, y_top), (x1 + w + 2, y_bottom), box_color, thickness=-1)
            cv2.putText(frame, line, (x1, y0 - 2), font, font_scale, (0,0,0), font_thick, cv2.LINE_AA)
            y0 = y_top  # next line above

        # Optional: draw center point
        cv2.circle(frame, (cx, cy), 3, box_color, -1)
    return frame

def get_vehicle_counts_in_roi(tracks, polygon, track_states):
    total_in_roi = 0
    class_counts = defaultdict(int)
    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        if is_center_in_roi(x1, y1, x2, y2, polygon):
            total_in_roi += 1
            state = track_states.get(track.track_id, {})
            cls_name = state.get('class_name', 'Unknown')
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

def log_data_to_csv(log_df, csv_file="./data/traffic_log.csv"):
    if not os.path.exists(csv_file):
        log_df.to_csv(csv_file, index=False)
    else:
        log_df.to_csv(csv_file, mode='a', header=False, index=False)

def write_live_traffic_data(total_in_roi, congestion_class, timestamp, csv_file="./data/traffic_live_data.csv"):
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
_mapping_df = pd.read_csv("./data/od_routes_mapping.csv")
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

def write_cnn_output_for_lstm(route, vehicle_count, congestion_level, timestamp, csv_file="./data/cnn_output_for_lstm.csv"):
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
    route: r"./videos/Traffic4.mp4"
    for route in _ROUTE_TO_OD.keys()
}

data_lock = threading.Lock()
latest_data = {}

def camera_worker(cam_id, src, model, tracker, threshold_medium, threshold_heavy,
                  roi_polygon=None, use_class_color=True, use_track_color=False,
                  custom_colors=None, max_frames=None):
    cap = cv2.VideoCapture(src)
    # Read FPS once for pacing; fallback to 30
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Prepare class colors once if you have class_names
    class_colors = generate_class_colors(class_names)

    # Per-track state
    track_states = {}  # track_id -> dict

    frame_count = 0

    if roi_polygon is None:
        roi_polygon = ROI_POLYGON

    while cap.isOpened():
        if max_frames is not None and frame_count >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 360))
        t0 = time.time()
        # Detection
        results = model(frame)[0]
        detections = []
        if results.boxes is not None and results.boxes.data is not None:
            for det in results.boxes.data:
                x1, y1, x2, y2, conf, cls = det.tolist()
                if conf < 0.8:
                    continue
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                cls = int(cls)
                # class_names loaded from YAML
                if cls < len(class_names):
                    class_name = class_names[cls]
                else:
                    class_name = str(cls)
                detections.append([[x1, y1, x2 - x1, y2 - y1], float(conf), cls, class_name])

        # Update tracks
        tracks = tracker.update_tracks(detections, frame=frame)

        # Attach class info for newly confirmed tracks
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            # If state not exist or missing class, attempt IoU match for this frame
            state = track_states.get(tid, {})
            if 'class_name' not in state or state.get('class_name') == 'Unknown':
                # Match by IoU
                l, t, r, b = track.to_ltrb()
                best_iou = 0.0
                best_det = None
                for det in detections:
                    x, y, w, h = det[0]
                    x2_, y2_ = x + w, y + h
                    iou = calculate_iou((l, t, r, b), det[0])
                    if iou > best_iou:
                        best_iou = iou
                        best_det = det
                if best_det and best_iou > 0.3:
                    cls_id = best_det[2]
                    cls_name = best_det[3]
                else:
                    cls_id = None
                    cls_name = 'Unknown'
                state.update({
                    'class_id': cls_id,
                    'class_name': cls_name,
                    # prev_center/time will be set in draw
                })
                track_states[tid] = state

        # Draw tracks (updates states: prev_center, prev_time, speed, direction)
        frame = draw_tracks(frame, tracks, track_states, class_colors,
                           use_class_color=use_class_color,
                           use_track_color=use_track_color,
                           custom_colors=custom_colors)

        # ROI drawing and counting as before
        cv2.polylines(frame, [roi_polygon], True, (255, 255, 0), 2)
        total_in_roi, class_counts_roi = get_vehicle_counts_in_roi(tracks, roi_polygon, track_states)
        congestion_class = get_congestion_level(total_in_roi, threshold_medium, threshold_heavy)
        timestamp = datetime.now().isoformat()
        write_cnn_output_for_lstm(cam_id, total_in_roi, congestion_class, timestamp)

        # Prepare speed_values and direction_counts from track_states of active tracks
        speed_values = []
        direction_counts = {"Up": 0, "Down": 0, "Left": 0, "Right": 0, "Static": 0, "Unknown": 0}
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            state = track_states.get(tid, {})
            sp = state.get('speed_m_s', 0.0)
            speed_values.append(sp)
            dir_str = state.get('direction', 'Unknown')
            direction_counts[dir_str] = direction_counts.get(dir_str, 0) + 1

        display_frame = frame[:, :, ::-1]
        with data_lock:
            latest_data[cam_id] = {
                "frame": display_frame,
                "total": total_in_roi,
                "congestion": congestion_class,
                "timestamp": timestamp,
                "vehicle_classes": class_counts_roi,
                "speed_values": speed_values,
                "direction_counts": direction_counts,
            }

        frame_count += 1

        # Sleep to roughly match FPS
        elapsed = time.time() - t0
        wait = max(0, (1.0 / fps) - elapsed)
        time.sleep(wait)

    cap.release()


import collections

def main():
    st.markdown(
        """
        <h1 style='text-align: center; color: gray; white-space: nowrap;'>
        <span style='color: #FF5733;'>Traffic Monitoring</span> Application
        </h1>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.header("Configuration & Controls")
    model_options = {
        "YOLOv8 (Default)": r"./models/yolov8_d5_100e.pt",
        "Custom YOLOv8 50 Epochs": r"./src/ckpts/yolov8_d5_50e.pt",
        "Custom YOLOv8 100 Epochs": r"./src/ckpts/yolov8_d5_100e.pt",
        "Custom YOLOv12 50 Epochs": r"./src/ckpts/yolov12_d5_50e.pt",
        "Custom YOLOv12 100 Epochs": r"./src/ckpts/yolov12_d5_100e.pt",
    }
    selected_model = st.sidebar.selectbox("YOLO Model Path", list(model_options.keys()), index=0)
    model_path = model_options[selected_model]
    threshold_medium = st.sidebar.slider("Medium Congestion Threshold", 1, 50, 8)
    threshold_heavy = st.sidebar.slider("Heavy Congestion Threshold", 5, 100, 15)
    # Add camera selection for user to view one camera at a time
    cam_ids = list(VIDEO_SOURCES.keys())
    selected_cam = st.sidebar.selectbox("Select Camera to View", cam_ids)

    # Add mode selection: Single camera (default) or Concurrent mode
    mode = st.sidebar.radio("Select Mode", ("Single Camera Mode (default)", "Concurrent Mode"))

    # Add ROI polygon adjustment UI
    st.sidebar.markdown("### Adjust ROI Polygon (IoU Box)")
    default_roi = ROI_POLYGON.tolist()
    roi_points = []
    for i, point in enumerate(default_roi):
        x = st.sidebar.slider(f"Point {i+1} X", 0, 1280, point[0])
        y = st.sidebar.slider(f"Point {i+1} Y", 0, 720, point[1])
        roi_points.append([x, y])
    roi_polygon = np.array(roi_points, dtype=np.int32)

    # Add bounding box color customization UI
    st.sidebar.markdown("### Bounding Box Color Customization")
    color_mode = st.sidebar.selectbox("Color Mode", ["Class Colors", "Track Colors", "Custom Colors"])
    custom_colors = None
    if color_mode == "Custom Colors":
        custom_colors = {}
        for idx, class_name in enumerate(class_names):
            default_color = generate_class_colors(class_names).get(idx, (0, 255, 0))
            default_hex = '#%02x%02x%02x' % (default_color[0], default_color[1], default_color[2])
            color_hex = st.sidebar.color_picker(f"Color for {class_name}", default_hex)
            # Convert hex to BGR tuple
            rgb = mcolors.hex2color(color_hex)
            bgr = tuple(int(c * 255) for c in rgb[::-1])
            custom_colors[idx] = bgr

    # Add max frames per chunk for video chunking
    max_frames = st.sidebar.number_input("Max Frames per Chunk (0 = full video)", min_value=0, max_value=10000, value=0, step=100)
    if max_frames == 0:
        max_frames = None

    # Data structures for visualization
    vehicle_count_history = collections.defaultdict(lambda: collections.deque(maxlen=100))
    congestion_history = collections.defaultdict(lambda: collections.deque(maxlen=100))
    speed_history = collections.defaultdict(lambda: collections.deque(maxlen=100))
    direction_history = collections.defaultdict(lambda: collections.deque(maxlen=100))

    if st.sidebar.button("Start Monitoring"):
        model, tracker = init_models(model_path)
        if mode == "Single Camera Mode (default)":
            # Start only the selected camera thread
            src = VIDEO_SOURCES[selected_cam]
            threading.Thread(
                target=camera_worker,
                args=(selected_cam, src, model, tracker, threshold_medium, threshold_heavy,
                      roi_polygon, color_mode=="Class Colors", color_mode=="Track Colors",
                      custom_colors, max_frames),
                daemon=True,
            ).start()
        else:
            # Start all camera threads concurrently
            for cam_id, src in VIDEO_SOURCES.items():
                threading.Thread(
                    target=camera_worker,
                    args=(cam_id, src, model, tracker, threshold_medium, threshold_heavy,
                          roi_polygon, color_mode=="Class Colors", color_mode=="Track Colors",
                          custom_colors, max_frames),
                    daemon=True,
                ).start()

    placeholders = {cam_id: st.empty() for cam_id in VIDEO_SOURCES}
    chart_placeholder = st.empty()

    while True:
        with data_lock:
            snapshot = latest_data.copy()
        # Update histories for selected camera
        if selected_cam in snapshot:
            data = snapshot[selected_cam]
            vehicle_count_history[selected_cam].append((data["timestamp"], data["total"]))
            congestion_history[selected_cam].append((data["timestamp"], data["congestion"]))
            # For speed and direction, we need to extract from tracks - not currently stored, so skipping for now
            if data:
                speed_history[selected_cam].append((data["timestamp"], data["speed_values"]))
                direction_history[selected_cam].append((data["timestamp"], data["direction_counts"]))

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

        # Visualizations
        with chart_placeholder.container():
            st.markdown("### Vehicle Count Over Time")
            if vehicle_count_history[selected_cam]:
                df_vc = pd.DataFrame(vehicle_count_history[selected_cam], columns=["Timestamp", "Count"])
                df_vc["Timestamp"] = pd.to_datetime(df_vc["Timestamp"])
                chart = alt.Chart(df_vc).mark_line().encode(
                    x="Timestamp:T",
                    y="Count:Q"
                ).properties(width=700, height=200)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.write("No vehicle count data yet.")

            st.markdown("### Congestion Level Over Time")
            if congestion_history[selected_cam]:
                df_cong = pd.DataFrame(congestion_history[selected_cam], columns=["Timestamp", "Level"])
                df_cong["Timestamp"] = pd.to_datetime(df_cong["Timestamp"])
                # Map congestion levels to numeric for charting
                level_map = {"Light": 1, "Medium": 2, "Heavy": 3}
                df_cong["LevelNum"] = df_cong["Level"].map(level_map)
                chart_cong = alt.Chart(df_cong).mark_line().encode(
                    x="Timestamp:T",
                    y="LevelNum:Q",
                    tooltip=["Level"]
                ).properties(width=700, height=200)
                st.altair_chart(chart_cong, use_container_width=True)
            else:
                st.write("No congestion data yet.")

            # Speed Distribution Visualization
            st.markdown("### Speed Distribution")
            if speed_history[selected_cam]:
                # Flatten speed values list for histogram
                all_speeds = []
                for _, speeds in speed_history[selected_cam]:
                    all_speeds.extend(speeds)
                if all_speeds:
                    df_speed = pd.DataFrame({"Speed (m/s)": all_speeds})
                    chart_speed = alt.Chart(df_speed).mark_bar().encode(
                        alt.X("Speed (m/s):Q", bin=alt.Bin(maxbins=30)),
                        y='count()',
                    ).properties(width=700, height=200)
                    st.altair_chart(chart_speed, use_container_width=True)
                else:
                    st.write("No speed data available.")
            else:
                st.write("No speed data yet.")

            # Directional Flow Diagram Visualization
            st.markdown("### Directional Flow Diagram")
            if direction_history[selected_cam]:
                # Aggregate direction counts over history
                total_direction_counts = {"Up": 0, "Down": 0, "Left": 0, "Right": 0, "Unknown": 0}
                for _, dir_counts in direction_history[selected_cam]:
                    for key in total_direction_counts.keys():
                        total_direction_counts[key] += dir_counts.get(key, 0)
                # Prepare data for bar chart
                df_dir = pd.DataFrame({
                    "Direction": list(total_direction_counts.keys()),
                    "Count": list(total_direction_counts.values())
                })
                chart_dir = alt.Chart(df_dir).mark_bar().encode(
                    x="Direction:N",
                    y="Count:Q",
                    color="Direction:N"
                ).properties(width=700, height=200)
                st.altair_chart(chart_dir, use_container_width=True)
            else:
                st.write("No direction data yet.")

        time.sleep(1)

if __name__ == "__main__":
    main()
