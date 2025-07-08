import threading
import time
import streamlit as st
from streamlit_lottie import st_lottie
import numpy as np
import sys
import os

import pickle
import folium
from streamlit_folium import folium_static
import pandas as pd
import heapq
import requests
import warnings
import random
from datetime import datetime, timedelta
import altair as alt
import plotly.express as px

from lstm_module import (
    run_lstm_predictions_on_cnn_output,
    update_graph_weights_with_lstm_predictions,
    predict_congestion_and_time,
    lstm_model,
    scaler_continuous,
    le_weather,
    le_origin,
    le_destination,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 1) Configure Chrome to run headlessly
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(options=options)

try:
    # 2) Navigate to your local Streamlit app
    driver.get("http://localhost:8501")

    # 3) Wait until the header (with class "stAppHeader") appears
    wait = WebDriverWait(driver, 10)
    header = wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".stAppHeader"))
    )

    # 4) Read its computed background-color
    #
    #    You can also try "background" if you want the full shorthand (in case there is an image or gradient),
    #    but most Streamlit themes set a solid color, so "background-color" will give you a single rgba/hex.
    bg_color = header.value_of_css_property("background-color")


finally:
    driver.quit()

st.markdown(
    """
    <style>
    body {
        background-color: #f0f4f8;
        font-family: 'Poppins', sans-serif;
        color: #2c3e50;
    }
    .stApp {
        padding: 20px 40px;
        background-color: bg_color;
    }
    header, .css-18e3th9 {
        background-color: #1e3c72 !important;
        color: black !important;
    }
    .css-1d391kg {
        background-color: #ffffff !important;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 10px 25px rgba(46, 82, 152, 0.15);
    }
    .stButton>button {
        background: linear-gradient(90deg, #1e3c72, #4286f4);
        color: white;
        border-radius: 12px;
        padding: 12px 20px;
        font-weight: 700;
        font-size: 16px;
        box-shadow: 0 6px 18px rgba(46, 82, 152, 0.3);
        transition: all 0.35s ease;
        width: 100%;
        margin-top: 10px;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #4286f4, #1e3c72);
        box-shadow: 0 8px 22px rgba(46, 82, 152, 0.45);
        transform: translateY(-3px);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1e3c72 !important;
        font-weight: 700 !important;
    }
    .stSelectbox label {
        color: #1e3c72 !important;
        font-weight: 700 !important;
        font-size: 18px !important;
    }
    .stSidebar .sidebar-content {
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
        padding: 20px;
        border-radius: 15px;
        font-family: 'Poppins', sans-serif;
        color: #2c3e50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function: load_graph_model
def load_graph_model(pickle_file):
    """
    Load a pre-trained graph model from a pickle file.
    """
    with open(pickle_file, 'rb') as f:
        graph = pickle.load(f)
    return graph

# Function: load_csv_dataset
def load_csv_dataset(csv_file):
    """
    Load a CSV dataset containing route and traffic data.
    Expected columns: Date, Time, Day of the Week, Origin, Destination, Route, Weather Conditions,
                      Accident Reports, Traffic Intensity, Distance.
    """
    data = pd.read_csv(csv_file)
    return data

# Function: modify_graph_with_data
def modify_graph_with_data(graph, data):
    """
    Modify the graph weights dynamically based on dataset information.
    Weight calculation uses distance, accident reports, and traffic intensity.
    """
    for index, row in data.iterrows():
        origin = row['Origin']
        destination = row['Destination']

        # Get required columns with defaults
        distance = row.get('Distance', 0)
        accidents = row.get('Accident Reports', 0)
        traffic = row.get('Traffic Intensity', 0)
        
        # Calculate weight: distance + (accidents factor) + (traffic factor)
        weight = distance + (accidents * 100) + (traffic * 10)
        
        # Ensure graph has nodes for origin and destination
        if origin not in graph:
            graph[origin] = {}
        if destination not in graph:
            graph[destination] = {}

        # Update the graph in both directions (undirected graph assumption)
        graph[origin][destination] = (weight, row.get('Route', 'N/A'))
        graph[destination][origin] = (weight, row.get('Route', 'N/A'))
        
    return graph

import os
import pandas as pd

def update_graph_weights_with_lstm_predictions(graph, lstm_predictions_file="./data/lstm_predictions_output.csv", prediction_horizon_minutes=10):
    """
    Update graph edge weights based on LSTM predicted traffic intensity for the selected prediction horizon.
    The LSTM predictions CSV should have columns: Route, Predicted Traffic Intensity, Timestamp, Weather Conditions, Accident Reports.
    Only predictions within the next prediction_horizon_minutes are considered.
    """
    import os
    import pandas as pd
    from datetime import datetime, timedelta

    if not os.path.exists(lstm_predictions_file):
        return graph  # No LSTM predictions, return original graph

    try:
        pred_df = pd.read_csv(lstm_predictions_file)
    except Exception:
        return graph  # On error, return original graph

    # Create 'Timestamp' column if missing by combining 'Date' and 'Time'
    if 'Timestamp' not in pred_df.columns and 'Date' in pred_df.columns and 'Time' in pred_df.columns:
        pred_df['Timestamp'] = pd.to_datetime(pred_df['Date'] + ' ' + pred_df['Time'])

    # Filter predictions within the prediction horizon from now
    now = datetime.now()
    horizon_end = now + timedelta(minutes=prediction_horizon_minutes)
    pred_df['Timestamp'] = pd.to_datetime(pred_df['Timestamp'])
    pred_df = pred_df[(pred_df['Timestamp'] >= now) & (pred_df['Timestamp'] <= horizon_end)]

    # Check if 'Predicted Traffic Intensity' column exists
    if 'Predicted Traffic Intensity' not in pred_df.columns:
        import logging
        logging.warning(f"'Predicted Traffic Intensity' column not found in {lstm_predictions_file}. Skipping LSTM weight update.")
        return graph

    # Aggregate predicted traffic intensity per route (mean)
    route_pred_map = pred_df.groupby('Route')['Predicted Traffic Intensity'].mean().to_dict()

    # Update graph weights for edges matching route names
    for origin in graph:
        for destination in graph[origin]:
            weight, route_name = graph[origin][destination]
            if route_name in route_pred_map:
                predicted_traffic = route_pred_map[route_name]
                # Adjust weight with predicted traffic intensity factor
                new_weight = weight + (predicted_traffic * 10)
                graph[origin][destination] = (new_weight, route_name)
            else:
                # No prediction for this route, keep original weight
                graph[origin][destination] = (weight, route_name)

    return graph

# Function: dijkstra
import math

def heuristic(node, end):
    """
    Heuristic function for A* algorithm: use straight-line distance if coordinates are available,
    otherwise return 0 (equivalent to Dijkstra).
    Assumes node and end are tuples of (latitude, longitude).
    """
    if not isinstance(node, tuple) or not isinstance(end, tuple):
        return 0
    # Haversine formula to calculate distance between two lat/lon points
    lat1, lon1 = node
    lat2, lon2 = end
    R = 6371  # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

import heapq
import math
import copy

def bidirectional_a_star(graph, start, end, location_coords=None):
    """
    Bidirectional A* search from start to end.
    Returns total cost, path as list, and route details.
    """
    def heuristic(node, goal):
        if not location_coords or node not in location_coords or goal not in location_coords:
            return 0
        lat1, lon1 = location_coords[node]
        lat2, lon2 = location_coords[goal]
        R = 6371  # Earth radius in km
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = math.sin(delta_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    # Initialize forward and backward search structures
    forward_queue = [(0 + heuristic(start, end), 0, start, [start], "")]
    backward_queue = [(0 + heuristic(end, start), 0, end, [end], "")]
    forward_visited = {}
    backward_visited = {}

    meeting_node = None
    best_cost = float("inf")
    best_path = []
    best_route = ""

    while forward_queue and backward_queue:
        # Forward step
        est_f, cost_f, current_f, path_f, route_f = heapq.heappop(forward_queue)
        if current_f in forward_visited:
            continue
        forward_visited[current_f] = (cost_f, path_f, route_f)
        if current_f in backward_visited:
            total_cost = cost_f + backward_visited[current_f][0]
            if total_cost < best_cost:
                best_cost = total_cost
                best_path = path_f + backward_visited[current_f][1][::-1][1:]
                best_route = route_f + backward_visited[current_f][2]
                meeting_node = current_f
        if current_f == end:
            break
        for neighbor, (weight, route) in graph.get(current_f, {}).items():
            if neighbor not in forward_visited:
                new_route_taken = route_f + f"{current_f} -> {neighbor} (Route: {route})\n"
                new_cost = cost_f + weight
                est_cost = new_cost + heuristic(neighbor, end)
                heapq.heappush(forward_queue, (est_cost, new_cost, neighbor, path_f + [neighbor], new_route_taken))

        # Backward step
        est_b, cost_b, current_b, path_b, route_b = heapq.heappop(backward_queue)
        if current_b in backward_visited:
            continue
        backward_visited[current_b] = (cost_b, path_b, route_b)
        if current_b in forward_visited:
            total_cost = cost_b + forward_visited[current_b][0]
            if total_cost < best_cost:
                best_cost = total_cost
                best_path = forward_visited[current_b][1] + path_b[::-1][1:]
                best_route = forward_visited[current_b][2] + route_b
                meeting_node = current_b
        if current_b == start:
            break
        for neighbor, (weight, route) in graph.get(current_b, {}).items():
            if neighbor not in backward_visited:
                new_route_taken = f"{neighbor} -> {current_b} (Route: {route})\n" + route_b
                new_cost = cost_b + weight
                est_cost = new_cost + heuristic(neighbor, start)
                heapq.heappush(backward_queue, (est_cost, new_cost, neighbor, path_b + [neighbor], new_route_taken))

    if best_path:
        return best_cost, best_path, best_route
    else:
        return float("inf"), [], ""

def dijkstra(graph, start, end, location_coords=None):
    """
    Wrapper to call bidirectional A* as improved pathfinding algorithm.
    """
    return bidirectional_a_star(graph, start, end, location_coords)

# Function: yen_k_shortest_paths
def yen_k_shortest_paths(graph, start, end, k=3):
    """
    Find up to k shortest paths between start and end nodes using Yen's algorithm.
    Returns a list of tuples: (total_cost, path, route_details).
    """
    import copy

    # First shortest path
    cost, path, route = dijkstra(graph, start, end)
    if not path:
        return []

    paths = [(cost, path, route)]
    potential_paths = []

    for i in range(1, k):
        for j in range(len(paths[-1][1]) - 1):
            spur_node = paths[-1][1][j]
            root_path = paths[-1][1][:j + 1]

            # Copy graph to modify
            graph_copy = copy.deepcopy(graph)

            # Remove edges that are part of previous paths sharing the same root path
            for p in paths:
                if len(p[1]) > j and p[1][:j + 1] == root_path:
                    u = p[1][j]
                    v = p[1][j + 1]
                    if v in graph_copy.get(u, {}):
                        del graph_copy[u][v]
                    if u in graph_copy.get(v, {}):
                        del graph_copy[v][u]

            # Remove nodes in root_path except spur_node to prevent loops
            # for node in root_path[:-1]:
            #     if node in graph_copy:
            #         graph_copy.pop(node)

            # Calculate spur path from spur_node to end
            spur_cost, spur_path, spur_route = dijkstra(graph_copy, spur_node, end)

            if spur_path and spur_path[0] == spur_node:
                total_path = root_path[:-1] + spur_path
                total_cost = 0
                total_route = ""
                for idx in range(len(total_path) - 1):
                    u = total_path[idx]
                    v = total_path[idx + 1]
                    weight, route_name = graph[u][v]
                    total_cost += weight
                    total_route += f"{u} -> {v} (Route: {route_name})\n"
                # Ensure origin and destination are included in route details explicitly
                origin = total_path[0]
                destination = total_path[-1]
                total_route = f"Origin: {origin}\nDestination: {destination}\nRoutes Taken:\n" + total_route
                potential_paths.append((total_cost, total_path, total_route))

        if not potential_paths:
            break

        # Sort potential paths by cost
        potential_paths.sort(key=lambda x: x[0])

        # Add the lowest cost path to paths
        paths.append(potential_paths.pop(0))

    # Add origin and destination info to the first path's route details as well
    if paths:
        origin = paths[0][1][0]
        destination = paths[0][1][-1]
        cost, path, route = paths[0]
        route = f"Origin: {origin}\nDestination: {destination}\nRoutes Taken:\n" + route
        paths[0] = (cost, path, route)

    return paths

# Function: load_lottieurl
def load_lottieurl(url: str):
    """
    Load a Lottie animation from a URL.
    """
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        return None


# Global variable to cache live traffic data
live_traffic_data = {
    "vehicles_in_roi": None,
    "congestion_level": None,
    "timestamp": None
}

# Function to read live traffic data from CSV periodically
def read_live_traffic_data(csv_file="./data/traffic_live_data.csv"):
    global live_traffic_data
    try:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            if not df.empty:
                # Read the last row (latest data)
                last_row = df.iloc[-1]
                live_traffic_data["vehicles_in_roi"] = last_row.get("vehicles_in_roi", None)
                live_traffic_data["congestion_level"] = last_row.get("congestion_level", None)
                live_traffic_data["timestamp"] = last_row.get("timestamp", None)
    except Exception as e:
        pass

# Start a background thread to update live traffic data every 5 seconds
def start_live_data_thread():
    def update_loop():
        while True:
            read_live_traffic_data()
            time.sleep(5)
    thread = threading.Thread(target=update_loop, daemon=True)
    thread.start()

start_live_data_thread()

def predict_congestion_and_time(route_data, time_adjustment=0):
    """
    Predict congestion level and travel time using LSTM model if available,
    otherwise fallback to rule-based prediction.
    Uses live traffic data from Traffic_Video.py if available.
    """
    # Use live traffic data if available
    if live_traffic_data["vehicles_in_roi"] is not None:
        traffic_intensity_live = float(live_traffic_data["vehicles_in_roi"])
    else:
        traffic_intensity_live = int(route_data['Traffic Intensity']) + time_adjustment

    if lstm_model is None or scaler_continuous is None or le_weather is None or le_origin is None or le_destination is None:
        # Fallback to rule-based prediction
        traffic_intensity = traffic_intensity_live
        if traffic_intensity < 30:
            congestion = 'Low'
            predicted_time = 15
        elif 30 <= traffic_intensity < 60:
            congestion = 'Medium'
            predicted_time = 30
        else:
            congestion = 'High'
            predicted_time = 60
        return congestion, predicted_time

    # Prepare input sequence for LSTM prediction
    try:
        origin_encoded = le_origin.transform([route_data['Origin']])[0]
        destination_encoded = le_destination.transform([route_data['Destination']])[0]
    except Exception:
        origin_encoded = 0
        destination_encoded = 0

    seq_length = 10
    input_seq = np.array([[origin_encoded, destination_encoded]] * seq_length, dtype=np.float32)
    X_input = input_seq.reshape((1, seq_length, 2))

    # Predict outputs
    route_pred, weather_pred, cont_pred = lstm_model.predict(X_input)

    # Inverse transform continuous predictions
    cont_unscaled = scaler_continuous.inverse_transform(cont_pred)

    # Extract predicted traffic intensity
    pred_traffic_intensity = cont_unscaled[0, 1]

    # Map predicted traffic intensity to congestion and time
    if pred_traffic_intensity < 30:
        congestion = 'Low'
        predicted_time = 15
    elif 30 <= pred_traffic_intensity < 60:
        congestion = 'Medium'
        predicted_time = 30
    else:
        congestion = 'High'
        predicted_time = 60

    return congestion, predicted_time

################################################################################
#                               APP CONFIGURATION                              #
################################################################################


# Back button in the top right corner
if st.button("Back", key='back_button'):
    st.session_state.page = 'home'  # Change session state to go back to home

# Custom CSS for select boxes and other components
st.markdown(
    """
    <style>
    .stSelectbox label {
        color: #333;
        font-weight: bold;
        font-size: 18px;
    }
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)
################################################################################
#                             LANGUAGE CONTROLS                                #
################################################################################
lang_mapping = {"vi":{
    "page_title": "Bảng điều khiển tắc nghẽn giao thông và tối ưu hóa lộ trình",
    # sidebar nav
    "side_nav_title": "Điều hướng & Điều khiển",
    "side_nav_data_file": "Đường dẫn tệp CSV dữ liệu",
    "side_nav_graph_file": "Đường dẫn tệp mô hình đồ thị",
    "side_nav_time_option": "Chọn thời gian trong ngày",
    "side_nav_time_option_values": ["Giờ cao điểm (8-10 AM, 5-7 PM)", "Giờ thấp điểm (Tất cả các thời gian khác)"],
    "side_nav_animation": "Hiện hoạt hình",
    # sidebar filtering
    "side_fil_title": "Lọc dữ liệu theo ngày",
    "side_fil_date_range": "Chọn khoảng thời gian",
    "side_fil_button": "XUẤT DỮ LIỆU ĐÃ LỌC",
    "side_fil_summary": "Hiện tóm tắt dữ liệu",
    # route prediction
    "page_title": "Bảng điều khiển tắc nghẽn giao thông và tối ưu hóa lộ trình",
    "select_route": "Chọn lộ trình của bạn",
    "select_origin_label": "Chọn điểm xuất phát",
    "select_destination_label": "Chọn điểm đến",
    "select_route_details": "Chi tiết lộ trình đã chọn từ dữ liệu lịch sử",
    "select_route_details_origin": "Điểm xuất phát",
    "select_route_details_destination": "Điểm đến",
    "route_prediction": "Dự đoán lộ trình",
    "route_prediction_congestion": "Mức độ tắc nghẽn",
    "route_prediction_time": "Thời gian di chuyển dự đoán",
    "route_prediction_weather": "Thời tiết hiện tại",
    "route_map": "Bản đồ lộ trình",
    "best_route": "Tối ưu hóa lộ trình tốt nhất bằng thuật toán Dijkstra",
    "best_route_details": "Lộ trình tốt nhất từ {} đến {} là:",
    "best_route_distance": f"**Khoảng cách (Chi phí):** {{:.2f}} đơn vị",
    "best_route_path": "Chi tiết lộ trình",
    # data exploration
    "data_exploration": "Khám phá và phân tích dữ liệu",
    "tabs_values": ["Tổng quan", "Phân tích giao thông", "Tai nạn & Thời tiết", "Xu hướng theo thời gian"],
    "tabs_overview": "Tổng quan",
    "tabs_traffic_analysis": "Phân tích giao thông",
    "tabs_accidents_weather": "Tình trạng tai nạn và thời tiết",
    "tabs_time_trends": "Xu hướng theo thời gian",
    # overview tab
    "overview_title": "Tổng quan về tập dữ liệu",
    "overview_sample": "Dưới đây là một mẫu của tập dữ liệu tắc nghẽn giao thông:",
    "overview_statistics": "Thống kê cơ bản",
    # traffic analysis tab
    "traffic_analysis_title": "Phân phối cường độ giao thông",
    "traffic_analysis_route": "Cường độ giao thông trung bình theo lộ trình",
    "traffic_analysis_distribution_title": "Phân phối cường độ giao thông",
    "traffic_analysis_distribution_x": "Cường độ giao thông",
    "traffic_analysis_distribution_y": "Số lượng",
    "traffic_analysis_x_label": "Lộ trình",
    "traffic_analysis_y_label": "Cường độ giao thông trung bình",
    # accidents & weather tab
    "accidents_weather_title": "Phân tích báo cáo tai nạn",
    "accidents_weather_weather": "Tổng quan về điều kiện thời tiết",
    "accidents_weather_distribution_title": "Phân phối báo cáo tai nạn",
    "accidents_weather_distribution_x": "Báo cáo tai nạn (Phân loại)",
    "accidents_weather_distribution_y": "Số lượng bản ghi",
    "accidents_weather_weather_conditions_title": "Tỷ lệ điều kiện thời tiết",
    "accidents_weather_weather_conditions": "Điều kiện thời tiết",
    "accidents_weather_weather_conditions_count": "Số lượng",
    # time trends tab
    "time_trends_title": "Xu hướng giao thông theo ngày trong tuần",
    "time_trends_by_day_graph_title": "Cường độ giao thông trung bình theo ngày trong tuần",
    "time_trends_by_day_graph_x": "Ngày trong tuần",
    "time_trends_accidents": "Tai nạn theo thời gian",
    "time_trends_accidents_graph_title": "Tổng số báo cáo tai nạn hàng ngày",
    "time_trends_accidents_graph_x": "Ngày",
    "time_trends_accidents_graph_y": "Báo cáo tai nạn",
    # interactive filters
    "interactive_filters": "Bộ lọc dữ liệu tương tác",
    "interactive_filters_sample": "Mẫu dữ liệu sau bộ lọc ngày",
    # advanced visualizations
    "advanced_visualizations": "Hình ảnh nâng cao",
    "advanced_scatter": "Biểu đồ phân tán khoảng cách so với cường độ giao thông",
    "advanced_scatter_x": "Khoảng cách",
    "advanced_scatter_y": "Cường độ giao thông",
    "advanced_heatmap": "Bản đồ nhiệt: Ngày trong tuần so với điều kiện thời tiết",
    "advanced_heatmap_title": "Bản đồ nhiệt cường độ giao thông trung bình theo ngày trong tuần và điều kiện thời tiết",
    "advanced_heatmap_x": "Ngày trong tuần",
    "advanced_heatmap_y": "Điều kiện thời tiết",
    "advanced_heatmap_z": "Tổng cường độ giao thông",
    # future improvements
    "future_improvements": "Cải tiến trong tương lai & Ghi chú của nhà phát triển",
    "future_improvements_note": """
- **Tích hợp dữ liệu thời gian thực:** Cân nhắc tích hợp nguồn dữ liệu giao thông trực tiếp để cập nhật dự đoán một cách động.
- **Mô hình học máy:** Dự đoán hiện tại dựa trên quy tắc. Tích hợp mô hình học máy được đào tạo trên dữ liệu lịch sử có thể cải thiện độ chính xác.
- **Tùy chỉnh của người dùng:** Mở rộng bộ lọc thanh bên để bao gồm lựa chọn lộ trình, điều kiện thời tiết và điều chỉnh thời gian trong ngày.
- **Cải tiến bản đồ:** Sử dụng dữ liệu địa lý chi tiết hơn và các điểm đánh dấu, và xem xét việc phân nhóm cho các khu vực đông đúc.
- **Tối ưu hóa hiệu suất:** Khi tập dữ liệu lớn lên, tối ưu hóa việc tải và lưu trữ dữ liệu để duy trì hiệu suất.
""",
    # footer
    "footer": """
---
**Bảng điều khiển tắc nghẽn giao thông và tối ưu hóa lộ trình**
Được phát triển bằng Streamlit, Folium, Altair và Plotly.
Liên hệ với nhóm phát triển để biết thêm chi tiết.
""",

    # additional utility code
    "data_summary": "Hiện tóm tắt dữ liệu",
    "data_summary_title": "Tóm tắt dữ liệu",
    "data_summary_records": "Số lượng bản ghi",
    "data_summary_columns": "Cột",
    
}, 
"en":{
    "page_title": "Traffic Congestion and Route Optimization Dashboard",
    # sidebar nav
    "side_nav_title": "Navigation & Controls",
    "side_nav_data_file": "Data CSV File Path",
    "side_nav_graph_file": "Graph Model File Path",
    "side_nav_time_option": "Select Time of Day",
    "side_nav_time_option_values": ["Peak Hours (8-10 AM, 5-7 PM)", "Off-Peak Hours (All other times)"],
    "side_nav_animation": "Show Animations",
    # sidebar filtering
    "side_fil_title": "Filter Data by Date",    
    "side_fil_date_range": "Select date range",
    "side_fil_button": "EXPORT FILTERED DATA",
    "side_fil_summary": "Show Data Summary",
    # route prediction
    "page_title": "Traffic Congestion and Route Optimization Dashboard",
    "select_route": "Select Your Route",
    "select_origin_label": "Select Origin",
    "select_destination_label": "Select Destination",
    "select_route_details": "Selected Route Details from Historical Data",
    "select_route_details_origin": "Origin",
    "select_route_details_destination": "Destination",
    "route_prediction": "Route Prediction",
    "route_prediction_congestion": "Congestion Level",
    "route_prediction_time": "Predicted Travel Time",
    "route_prediction_weather": "Current Weather",
    "route_map": "Route Map",
    "best_route": "Best Route Optimization Using Dijkstra's Algorithm",
    "best_route_details": "The best route from {} to {} is:",
    "best_route_distance": f"**Distance (Cost): {{:.2f}} units**",
    "best_route_path": "Detailed Route Path",
    # data exploration
    "data_exploration": "Data Exploration & Analysis",
    "tabs_values": ["Overview", "Traffic Analysis", "Accidents & Weather", "Time Trends"],
    "tabs_overview": "Overview",
    "tabs_traffic_analysis": "Traffic Analysis",
    "tabs_accidents_weather": "Accidents & Weather",
    "tabs_time_trends": "Time Trends",
    # overview tab
    "overview_title": "Dataset Overview",
    "overview_sample": "Below is a sample of the traffic congestion dataset:",
    "overview_statistics": "Basic Statistics",
    # traffic analysis tab
    "traffic_analysis_title": "Traffic Intensity Distribution",
    "traffic_analysis_route": "Average Traffic Intensity by Route",
    "traffic_analysis_distribution_title": "Distribution of Traffic Intensity",
    "traffic_analysis_distribution_x": "Traffic Intensity",
    "traffic_analysis_distribution_y": "Count",
    "traffic_analysis_x_label": "Route",
    "traffic_analysis_y_label": "Avg Traffic Intensity",
    # accidents & weather tab
    "accidents_weather_title": "Accident Reports Analysis",
    "accidents_weather_weather": "Weather Conditions Overview",
    "accidents_weather_distribution_title": "Distribution of Accident Reports",
    "accidents_weather_distribution_x": "Accident Reports (Binned)",
    "accidents_weather_distribution_y": "Count of Records",
    "accidents_weather_weather_conditions_title": "Weather Conditions Proportion",
    "accidents_weather_weather_conditions": "Weather Conditions",
    "accidents_weather_weather_conditions_count": "Count",
    # time trends tab
    "time_trends_title": "Traffic Trends by Day of the Week",
    "time_trends_by_day_graph_title": "Average Traffic Intensity by Day of the Week",
    "time_trends_by_day_graph_x": "Day of the Week",
    "time_trends_by_day_graph_y": "Traffic Intensity",
    "time_trends_accidents": "Accidents Over Time",
    "time_trends_accidents_graph_title": "Daily Total Accident Reports",
    "time_trends_accidents_graph_x": "Date",
    "time_trends_accidents_graph_y": "Accident Reports",
    # interactive filters
    "interactive_filters": "Interactive Data Filters",
    "interactive_filters_sample": "Data Sample After Date Filter",
    # advanced visualizations
    "advanced_visualizations": "Advanced Visualizations",
    "advanced_scatter": "Distance vs. Traffic Intensity Scatter Plot",
    "advanced_scatter_x": "Distance",
    "advanced_scatter_y": "Traffic Intensity",
    "advanced_heatmap": "Heatmap: Day of the Week vs Weather Conditions",
    "advanced_heatmap_title": "Average Traffic Intensity Heatmap by Day of the Week and Weather Conditions",
    "advanced_heatmap_x": "Day of the Week",
    "advanced_heatmap_y": "Weather Conditions",
    "advanced_heatmap_z": "Sum of Traffic Intensity",
    # future improvements
    "future_improvements": "Future Improvements & Developer Notes",
    "future_improvements_note": """
- **Real-time Data Integration:** Consider integrating live traffic data feeds to update predictions dynamically.
- **Machine Learning Models:** The current prediction is rule-based. Integrating an ML model trained on historical data could improve accuracy.
- **User Customization:** Expand sidebar filters to include route selection, weather conditions, and time-of-day adjustments.
- **Mapping Enhancements:** Use more detailed geospatial data and markers, and consider clustering for dense areas.
- **Performance Optimization:** As the dataset grows, optimize data loading and caching to maintain performance.
""",
    # footer
    "footer": """
---
**Traffic Congestion and Route Optimization Dashboard**
Developed using Streamlit, Folium, Altair, and Plotly.
For more details, contact the development team.
""",
    # additional utility code
    "data_summary_title": "Data Summary",
    "data_summary_records": "Number of records",
    "data_summary_columns": "Columns",
}}

if "lang" not in st.session_state:
    st.session_state.lang = "vi"  # Default to Vietnamese

lang = st.session_state.lang
################################################################################
#                             SIDEBAR CONTROLS                                 #
################################################################################

st.sidebar.header(lang_mapping[lang]["side_nav_title"])

# Sidebar: Data file selection (for extensibility)
data_file = st.sidebar.text_input(lang_mapping[lang]["side_nav_data_file"], "./data/lstm_predictions_output.csv")
graph_file = st.sidebar.text_input(lang_mapping[lang]["side_nav_graph_file"], "./models/route_optimization_graph.pkl")

# Sidebar: Time of Day selection for prediction adjustments
time_option = st.sidebar.selectbox(
    lang_mapping[lang]["side_nav_time_option"],
    lang_mapping[lang]["side_nav_time_option_values"],
)

# Sidebar: Animation toggle
animation_toggle = st.sidebar.checkbox("Show Animations", value=True)

################################################################################
#                              LOAD DATA & MODELS                              #
################################################################################

def clean_date_column(df, date_col="Date"):
    """
    Clean the date column by coercing errors and dropping invalid dates.
    """
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    return df

# Load traffic dataset
try:
    traffic_data = load_csv_dataset(data_file)
    # Clean 'Date' column robustly
    traffic_data = clean_date_column(traffic_data, "Date")
except Exception as e:
    st.error(f"Error loading CSV file: {e}")
    st.stop()

# Extract unique route entries (drop duplicates)
unique_routes = traffic_data[['Origin', 'Destination', 'Route', 'Weather Conditions', 'Accident Reports', 'Traffic Intensity', 'Distance']].drop_duplicates()

# Load graph model
try:
    graph = load_graph_model(graph_file)
except Exception as e:
    st.error(f"Error loading graph model: {e}")
    st.stop()

# Modify graph with real-time dataset information
graph = modify_graph_with_data(graph, unique_routes)

# Sidebar: Prediction horizon selection for LSTM predictions
prediction_horizon = st.sidebar.slider("Select Prediction Horizon (minutes)", min_value=10, max_value=60, value=10, step=10)

# Update graph weights with LSTM predicted traffic data for selected horizon
graph = update_graph_weights_with_lstm_predictions(graph, lstm_predictions_file="./data/lstm_predictions_output.csv", prediction_horizon_minutes=prediction_horizon)

import threading
import time
import glob

# Adjust cnn_output_file pattern to match actual files generated by Traffic_Video.py
cnn_output_files = glob.glob("./data/cnn_output_for_lstm.csv")
if not cnn_output_files:
    import logging
    logging.warning("No CNN output files found matching pattern './data/cnn_output_for_lstm.csv'. LSTM predictions will be skipped.")

def lstm_prediction_update_loop():
    while True:
        run_lstm_predictions_on_cnn_output(cnn_output_file=cnn_output_files[0] if cnn_output_files else "./data/cnn_output_for_lstm.csv")
        time.sleep(10)

lstm_thread = threading.Thread(target=lstm_prediction_update_loop, daemon=True)
lstm_thread.start()

################################################################################
#                               TITLE & HEADER                                 #
################################################################################

st.markdown(
    """
    <h1 style='text-align: center; color: #2c3e50;'>
        {page_title}
    </h1>
    """.format(page_title=lang_mapping[lang]["page_title"]),
    unsafe_allow_html=True
)

################################################################################
#                             ANIMATION DISPLAY                                #
################################################################################

if animation_toggle:
    # Load traffic animation
    traffic_animation = load_lottieurl("https://lottie.host/ac9b87c1-302c-4eae-89af-1ac6d54c163b/8gYZJC5OpB.json")
    if traffic_animation:
        st_lottie(traffic_animation, height=200, key="traffic")

################################################################################
#                           WEATHER ANIMATION & INFO                           #
################################################################################

def get_weather_animation():
    """
    Randomly select a weather animation (sunny or cloudy) based on the current day.
    """
    current_day = datetime.now().day
    random.seed(current_day)
    weather_choice = random.choice(['clear', 'cloudy'])
    if weather_choice == 'clear':
        return load_lottieurl("https://lottie.host/d8ca6425-9e61-49b9-9bf7-8247dac3a459/tJrOUe25Kc.json"), "Sunny"
    else:
        return load_lottieurl("https://lottie.host/a6de97d2-f147-4699-9a80-e1f6aba6bb6d/FZJoVUa2Np.json"), "Cloudy"

weather_animation, weather_description = get_weather_animation()

################################################################################
#                          USER INPUT FOR ROUTE SELECTION                      #
################################################################################

st.subheader(lang_mapping[lang]["select_route"])

location_coords = {
    "Chợ Bình Tây": (10.749827770462066, 106.65103751151939),
    "Đền Bà Thiên Hậu": (10.75336599156594, 106.66131566993364),
    "Chợ Bình Đông": (10.740305157329562, 106.64346801433803),
    "Bưu điện Chợ Lớn": (10.75015281650651, 106.65920920671381),
    "Trung tâm Văn hóa Chợ Lớn": (10.752567211813698, 106.66837345199895),
    "Bệnh viện Quận 5": (10.754172214677302, 106.6658424815772),
    "Trường Tiểu học Chợ Lớn": (10.75146329767459, 106.6653704125339),
    "Trung tâm Thương Mại Chợ Lớn": (10.754380689267403, 106.66551845199916),
    "Đền Vua": (10.753735421388246, 106.68254348391028),
    "Trường Trung học Chợ Lớn": (10.752469890339768, 106.66679354438318),
    "Nhà thờ Chợ Lớn": (10.752450409355223, 106.65462434250949),
    "Công viên District 5": (10.765354741619404, 106.68074468391033),
    "Trung tâm Y tế Quận 5": (10.75973350653036, 106.669339853849),
    "Trung tâm Văn hóa Dân gian": (10.752621903536784, 106.66833300315976),
    "Sân vận động Quận 5": (10.76126477470413, 106.66344058113613),
    "Bảo tàng Quận 5": (10.776855424333268, 106.69992342910145),
    "Ngã tư Lê Văn Sỹ": (10.79663069590259, 106.66568831496896),
    "Trạm xe buýt Quận 5": (10.762648542663369, 106.67774630409893),
    "Cửa hàng Điện máy X": (10.755582614035308, 106.68065572575534),
    "Siêu thị Co-op Mart": (10.761366423779116, 106.67140083683259)
    }

# Filter origins to only those present in location_coords
valid_origins = [o for o in unique_routes['Origin'].unique() if o in location_coords]

# Dropdown: Select Origin with filtered valid origins
origin = st.selectbox(lang_mapping[lang]["select_origin_label"], valid_origins)

# Filter destinations based on origin and location_coords
def get_destinations(origin):
    dests = unique_routes[unique_routes['Origin'] == origin]['Destination'].unique()
    # Filter destinations to only those with valid location coordinates
    valid_dests = [d for d in dests if d in location_coords]
    return valid_dests

destination = st.selectbox(lang_mapping[lang]["select_destination_label"]
    , get_destinations(origin))

# Display selected date/time info from dataset sample (if desired)
st.write(f"### {lang_mapping[lang]['select_route_details']}")
st.write(f"**{lang_mapping[lang]['select_route_details_origin']}:** {origin}")
st.write(f"**{lang_mapping[lang]['select_route_details_destination']}:** {destination}")

################################################################################
#                        ROUTE PREDICTION & CONGESTION INFO                    #
################################################################################

# Get selected route data (if multiple, choose first)
try:
    selected_route = unique_routes[(unique_routes['Origin'] == origin) & 
                                   (unique_routes['Destination'] == destination)].iloc[0]
except IndexError:
    st.error("No data available for the selected route. Please try another combination.")
    st.stop()

# Adjust traffic intensity based on time selection
time_adjustment = 20 if time_option == (
    "Peak Hours (8-10 AM, 5-7 PM)" if lang == "en" else "Giờ cao điểm (8-10 AM, 5-7 PM)"
) else 0

# Predict congestion and travel time
congestion, predicted_time = predict_congestion_and_time(selected_route, time_adjustment)

# Display prediction details
st.markdown(f"#### {lang_mapping[lang]['route_prediction']}")
st.write(f"**{lang_mapping[lang]['route_prediction_congestion']}:** {congestion}")
st.write(f"**{lang_mapping[lang]['route_prediction_time']}:** {predicted_time} minutes")
st.write(f"**{lang_mapping[lang]['route_prediction_weather']}:** {weather_description}")
if animation_toggle and weather_animation:
    st_lottie(weather_animation, height=200, key="weather")

################################################################################
#                         MAP VISUALIZATION SECTION                            #
################################################################################

# Coordinates dictionary for map visualization
# Note: For District 5, update these with real coordinates of your landmarks
location_coords = {
    "Chợ Bình Tây": (10.749827770462066, 106.65103751151939),
    "Đền Bà Thiên Hậu": (10.75336599156594, 106.66131566993364),
    "Chợ Bình Đông": (10.740305157329562, 106.64346801433803),
    "Bưu điện Chợ Lớn": (10.75015281650651, 106.65920920671381),
    "Trung tâm Văn hóa Chợ Lớn": (10.752567211813698, 106.66837345199895),
    "Bệnh viện Quận 5": (10.754172214677302, 106.6658424815772),
    "Trường Tiểu học Chợ Lớn": (10.75146329767459, 106.6653704125339),
    "Trung tâm Thương Mại Chợ Lớn": (10.754380689267403, 106.66551845199916),
    "Đền Vua": (10.753735421388246, 106.68254348391028),
    "Trường Trung học Chợ Lớn": (10.752469890339768, 106.66679354438318),
    "Nhà thờ Chợ Lớn": (10.752450409355223, 106.65462434250949),
    "Công viên District 5": (10.765354741619404, 106.68074468391033),
    "Trung tâm Y tế Quận 5": (10.75973350653036, 106.669339853849),
    "Trung tâm Văn hóa Dân gian": (10.752621903536784, 106.66833300315976),
    "Sân vận động Quận 5": (10.76126477470413, 106.66344058113613),
    "Bảo tàng Quận 5": (10.776855424333268, 106.69992342910145),
    "Ngã tư Lê Văn Sỹ": (10.79663069590259, 106.66568831496896),
    "Trạm xe buýt Quận 5": (10.762648542663369, 106.67774630409893),
    "Cửa hàng Điện máy X": (10.755582614035308, 106.68065572575534),
    "Siêu thị Co-op Mart": (10.761366423779116, 106.67140083683259)
}

# Retrieve lat/lon for origin and destination
if origin not in location_coords or destination not in location_coords:
    st.error("Origin or destination coordinates not found in location data. Please verify the location names.")
    lat_o, lon_o = None, None
    lat_d, lon_d = None, None
else:
    lat_o, lon_o = location_coords[origin]
    lat_d, lon_d = location_coords[destination]

import openrouteservice
from openrouteservice import convert

def get_route_coordinates(client, start_coords, end_coords):
    try:
        routes = client.directions(coordinates=[start_coords, end_coords], profile='driving-car', format='geojson')
        geometry = routes['features'][0]['geometry']
        coords = geometry['coordinates']
        # Convert from [lon, lat] to [lat, lon] for folium
        route_coords = [(coord[1], coord[0]) for coord in coords]
        return route_coords
    except Exception as e:
        st.error(f"Error fetching route from OpenRouteService: {e}")
        if lat_o is not None and lat_d is not None:
            return [(lat_o, lon_o), (lat_d, lon_d)]
        else:
            return []

if lat_o is None or lon_o is None or lat_d is None or lon_d is None:
    # Do not proceed with map if coordinates missing
    pass
else:
    polyline_color = "green" if congestion == "Low" else "yellow" if congestion == "Medium" else "red"
    m = folium.Map(location=[lat_o, lon_o], zoom_start=13)

    # Initialize OpenRouteService client with API key (replace 'YOUR_ORS_API_KEY' with actual key)
    ors_client = openrouteservice.Client(key='5b3ce3597851110001cf6248ff4d50d1b1f2404e90d76d6eec09b509')

    # Get route coordinates snapped to roads
    route_coords = get_route_coordinates(ors_client, (lon_o, lat_o), (lon_d, lat_d))

    folium.PolyLine(
        locations=route_coords,
        color=polyline_color,
        weight=6,
        tooltip=f"{origin} to {destination}: {congestion} congestion"
    ).add_to(m)

    # Add markers for all locations in location_coords
    for loc_name, (lat, lon) in location_coords.items():
        folium.Marker(
            location=[lat, lon],
            popup=loc_name,
            icon=folium.Icon(icon='info-sign', color='blue')
        ).add_to(m)

    # Add camera nodes with video playback in popup
    camera_nodes = {
    "Camera 1": {
        "coords": (10.752, 106.660),
        "video_url": "https://www.mapbox.com/bites/00188/patricia_nasa.webm"
    },
    "Camera 2": {
        "coords": (10.755, 106.670),
        "video_url": "https://www.mapbox.com/bites/00188/patricia_nasa.webm"
    },
    "Camera 3": {
        "coords": (10.760, 106.665),
        "video_url": "https://www.mapbox.com/bites/00188/patricia_nasa.webm"
    },
    "Camera 4": {
        "coords": (10.748, 106.655),
        "video_url": "https://www.mapbox.com/bites/00188/patricia_nasa.webm"
    },
    "Camera 5": {
        "coords": (10.754, 106.675),
        "video_url": "https://www.mapbox.com/bites/00188/patricia_nasa.webm"
    },
    "Camera 6": {
        "coords": (10.758, 106.660),
        "video_url": "https://www.mapbox.com/bites/00188/patricia_nasa.webm"
    },
    "Camera 7": {
        "coords": (10.763, 106.668),
        "video_url": "https://www.mapbox.com/bites/00188/patricia_nasa.webm"
    },
    "Camera 8": {
        "coords": (10.750, 106.662),
        "video_url": "https://www.mapbox.com/bites/00188/patricia_nasa.webm"
    },
    "Camera 9": {
        "coords": (10.757, 106.670),
        "video_url": "https://www.mapbox.com/bites/00188/patricia_nasa.webm"
    },
    "Camera 10": {
        "coords": (10.755, 106.665),
        "video_url": "https://www.mapbox.com/bites/00188/patricia_nasa.webm"
    },
    "Camera 11": {
        "coords": (10.759, 106.672),
        "video_url": "https://www.mapbox.com/bites/00188/patricia_nasa.webm"
    },
    "Camera 12": {
        "coords": (10.761, 106.658),
        "video_url": "https://www.mapbox.com/bites/00188/patricia_nasa.webm"
    },
    "Camera 13": {
        "coords": (10.753, 106.664),
        "video_url": "https://www.mapbox.com/bites/00188/patricia_nasa.webm"
    },
    "Camera 14": {
        "coords": (10.756, 106.669),
        "video_url": "https://www.mapbox.com/bites/00188/patricia_nasa.webm"
    },
    "Camera 15": {
        "coords": (10.758, 106.663),
        "video_url": "https://www.mapbox.com/bites/00188/patricia_nasa.webm"
    }
    }

    for cam_name, cam_info in camera_nodes.items():
        video_html = f'''
            <video width="320" height="250" controls autoplay muted loop playsinline>
                <source src="{cam_info["video_url"]}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        '''
        iframe = folium.IFrame(html=video_html, width=330, height=270)
        popup = folium.Popup(iframe, max_width=330)
        folium.Marker(
            location=cam_info["coords"],
            popup=popup,
            icon=folium.Icon(icon='camera', prefix='fa', color='red')
        ).add_to(m)

    folium.CircleMarker(
        location=[lat_o, lon_o],
        radius=8,
        color="blue",
        fill=True,
        fill_color="blue",
        fill_opacity=0.7,
        popup=f"Start: {origin}"
    ).add_to(m)
    folium.CircleMarker(
        location=[lat_d, lon_d],
        radius=8,
        color="green",
        fill=True,
        fill_color="green",
        fill_opacity=0.7,
        popup=f"End: {destination}"
    ).add_to(m)
    st.subheader(lang_mapping[lang]["route_map"])
    folium.Marker(location=[lat_o, lon_o], popup=origin).add_to(m)
    folium.Marker(location=[lat_d, lon_d], popup=destination, icon=folium.Icon(color="green")).add_to(m)
    # Increase map size and fix text color by setting width and height in folium_static
    folium_static(m, width=1500, height=500)

################################################################################
#                          BEST ROUTE OPTIMIZATION                             #
################################################################################

st.markdown(f"## {lang_mapping[lang]['best_route']}")
if origin and destination:
    # Check if origin and destination exist in graph
    if origin not in graph or destination not in graph:
        st.error("Origin or destination not found in the graph data.")
    else:
        # Get up to 3 alternative routes using Yen's algorithm
        k = 3
        routes = yen_k_shortest_paths(graph, origin, destination, k)

        if routes:
            # UI to select route among alternatives
            route_options = [f"Route {i+1} - Cost: {round(r[0], 2)}" for i, r in enumerate(routes)]
            selected_route_index = st.selectbox("Select Route Option", options=range(len(routes)), format_func=lambda x: route_options[x])

            selected_route = routes[selected_route_index]
            total_cost, path, route_taken = selected_route

            # Extract origin and destination from route_taken string if present
            origin_in_route = None
            destination_in_route = None
            route_lines = route_taken.splitlines()
            for line in route_lines:
                if line.startswith("Origin:"):
                    origin_in_route = line.replace("Origin:", "").strip()
                elif line.startswith("Destination:"):
                    destination_in_route = line.replace("Destination:", "").strip()

            # Fallback to UI selected origin/destination if not found in route details
            display_origin = origin_in_route if origin_in_route else origin
            display_destination = destination_in_route if destination_in_route else destination

            st.write(f"**{lang_mapping[lang]['best_route_details'].format(display_origin, display_destination)}**")
            st.write(f"**{lang_mapping[lang]['best_route_distance'].format(total_cost)}**")
            st.text(f"{lang_mapping[lang]['best_route_path']}")
            st.text(route_taken)

            # Show selected route on map
            # Instead of just node coords, get full route geometry from OpenRouteService for smooth road path
            try:
                # Prepare coordinates for ORS: list of [lon, lat]
                coords_for_ors = []
                for node in path:
                    if node in location_coords:
                        lat, lon = location_coords[node]
                        coords_for_ors.append([lon, lat])
                    else:
                        # If any node missing coords, fallback to previous method
                        coords_for_ors = []
                        break

                if coords_for_ors and len(coords_for_ors) >= 2:
                    # Get route geometry from ORS
                    ors_route = ors_client.directions(coordinates=coords_for_ors, profile='driving-car', format='geojson')
                    geometry = ors_route['features'][0]['geometry']
                    coords = geometry['coordinates']
                    # Convert to [lat, lon] for folium
                    path_coords = [(c[1], c[0]) for c in coords]
                    # Calculate cost from ORS summary if available
                    total_cost = ors_route['features'][0]['properties'].get('summary', {}).get('distance', total_cost)
                else:
                    # Fallback to previous method
                    path_coords = []
                    for i in range(len(path) - 1):
                        u = path[i]
                        v = path[i + 1]
                        if u in location_coords:
                            path_coords.append(location_coords[u])
                        if i == len(path) - 2 and v in location_coords:
                            path_coords.append(location_coords[v])
            except Exception as e:
                st.error(f"Error fetching route geometry from OpenRouteService: {e}")
                # Fallback to previous method
                path_coords = []
                for i in range(len(path) - 1):
                    u = path[i]
                    v = path[i + 1]
                    if u in location_coords:
                        path_coords.append(location_coords[u])
                    if i == len(path) - 2 and v in location_coords:
                        path_coords.append(location_coords[v])

            # Create folium map centered on origin
            m = folium.Map(location=location_coords.get(origin, [0, 0]), zoom_start=13)

            # Draw the selected route polyline
            folium.PolyLine(
                locations=path_coords,
                color="blue",
                weight=6,
                tooltip=f"Selected route: Cost {total_cost}"
            ).add_to(m)

            # Add markers for origin and destination
            folium.Marker(location=location_coords.get(origin, [0, 0]), popup=origin, icon=folium.Icon(color="green")).add_to(m)
            folium.Marker(location=location_coords.get(destination, [0, 0]), popup=destination, icon=folium.Icon(color="red")).add_to(m)

            folium_static(m, width=1500, height=500)

        else:
            if lang == "en":
                st.write("No path found between the selected origin and destination.")
            else:
                st.write("Không tìm thấy lộ trình giữa điểm xuất phát và điểm đến đã chọn.")

################################################################################
#                          DATA EXPLORATION & VISUALIZATION                    #
################################################################################

st.markdown(f"## {lang_mapping[lang]['data_exploration']}")

# Create tabs for various exploratory views
tabs = st.tabs(lang_mapping[lang]["tabs_values"])

# --------------------- TAB 1: OVERVIEW ---------------------
with tabs[0]:
    st.subheader(lang_mapping[lang]["overview_title"])
    st.write(lang_mapping[lang]["overview_sample"])
    st.dataframe(traffic_data.head(20), height=300)

    st.write(f"### {lang_mapping[lang]['overview_statistics']}")
    st.write(traffic_data.describe())

# --------------------- TAB 2: TRAFFIC ANALYSIS ---------------------
with tabs[1]:
    st.subheader(lang_mapping[lang]["traffic_analysis_title"])
    fig_traffic = px.histogram(traffic_data, x="Traffic Intensity", nbins=50,
                               title=lang_mapping[lang]["traffic_analysis_distribution_title"],
                                 labels={"Traffic Intensity": lang_mapping[lang]["traffic_analysis_distribution_x"],
                                            "count": lang_mapping[lang]["traffic_analysis_distribution_y"]})
    st.plotly_chart(fig_traffic, use_container_width=True)
    
    st.subheader(lang_mapping[lang]["traffic_analysis_route"])
    route_traffic = traffic_data.groupby("Route")["Traffic Intensity"].mean().reset_index()
    fig_route_traffic = px.bar(route_traffic, x="Route", y="Traffic Intensity",
                               title=lang_mapping[lang]["traffic_analysis_route"],
                               labels={"Traffic Intensity": lang_mapping[lang]["traffic_analysis_y_label"],
                                       "Route": lang_mapping[lang]["traffic_analysis_x_label"]})
    st.plotly_chart(fig_route_traffic, use_container_width=True)

# --------------------- TAB 3: ACCIDENTS & WEATHER ---------------------
with tabs[2]:
    st.subheader(lang_mapping[lang]["accidents_weather_title"])
    fig_accidents = alt.Chart(traffic_data).mark_bar().encode(
        x=alt.X("Accident Reports:Q", bin=alt.Bin(maxbins=20)),
        y='count()',
        tooltip=['count()']
    ).properties(width=700, height=400, title=lang_mapping[lang]["accidents_weather_distribution_title"])
    st.altair_chart(fig_accidents, use_container_width=True)

    st.subheader(lang_mapping[lang]["accidents_weather_weather"])
    weather_counts = traffic_data["Weather Conditions"].value_counts().reset_index()
    weather_counts.columns = [lang_mapping[lang]["accidents_weather_weather_conditions"],
                                lang_mapping[lang]["accidents_weather_weather_conditions_count"]]
    fig_weather = px.pie(weather_counts, names=lang_mapping[lang]["accidents_weather_weather_conditions"],
                            values=lang_mapping[lang]["accidents_weather_weather_conditions_count"],
                         title=lang_mapping[lang]["accidents_weather_weather_conditions_title"])
    st.plotly_chart(fig_weather, use_container_width=True)

# --------------------- TAB 4: TIME TRENDS ---------------------
with tabs[3]:
    st.subheader(lang_mapping[lang]["time_trends_title"])
    day_traffic = traffic_data.groupby("Day of the Week")["Traffic Intensity"].mean().reset_index()
    fig_day_traffic = px.line(day_traffic, x="Day of the Week", y="Traffic Intensity",
                              title=lang_mapping[lang]["time_trends_by_day_graph_title"],
                                labels={"Traffic Intensity": lang_mapping[lang]["traffic_analysis_y_label"],
                                        "Day of the Week": lang_mapping[lang]["time_trends_by_day_graph_x"]},
                              markers=True)
    st.plotly_chart(fig_day_traffic, use_container_width=True)

    st.subheader(lang_mapping[lang]["time_trends_accidents"])
    # Convert Date column to datetime with error coercion to handle invalid dates
    traffic_data["Date"] = pd.to_datetime(traffic_data["Date"], errors='coerce')
    # Drop rows with invalid dates
    traffic_data = traffic_data.dropna(subset=["Date"])
    accidents_over_time = traffic_data.groupby("Date")["Accident Reports"].sum().reset_index()
    fig_accidents_time = px.area(accidents_over_time, x="Date", y="Accident Reports",
                                 title="Daily Total Accident Reports",
                                    labels={"Accident Reports": lang_mapping[lang]["time_trends_accidents_graph_y"],
                                            "Date": lang_mapping[lang]["time_trends_accidents_graph_x"]})
    st.plotly_chart(fig_accidents_time, use_container_width=True)

################################################################################
#                          CUSTOM INTERACTIVE FILTERS                          #
################################################################################

st.markdown(f"## {lang_mapping[lang]['interactive_filters']}")

# Sidebar filter for date range selection
st.sidebar.markdown(f"### {lang_mapping[lang]['side_fil_title']}")
min_date = traffic_data["Date"].min()
max_date = traffic_data["Date"].max()
date_range = st.sidebar.date_input(lang_mapping[lang]["side_fil_date_range"], [min_date, max_date])

# Filter dataset based on selected date range
if len(date_range) == 2:
    filtered_data = traffic_data[(traffic_data["Date"] >= pd.to_datetime(date_range[0])) &
                                 (traffic_data["Date"] <= pd.to_datetime(date_range[1]))]
else:
    filtered_data = traffic_data.copy()

st.write(f"### {lang_mapping[lang]['interactive_filters_sample']}")
st.dataframe(filtered_data.head(10), height=250)

################################################################################
#                          ADVANCED VISUALIZATION                              #
################################################################################

st.markdown(f"## {lang_mapping[lang]['advanced_visualizations']}")

# Advanced Chart: Interactive Scatter Plot for Distance vs Traffic Intensity
st.subheader(lang_mapping[lang]["advanced_scatter"])
scatter_fig = px.scatter(filtered_data, x="Distance", y="Traffic Intensity",
                         color="Weather Conditions",
                         hover_data=["Origin", "Destination", "Accident Reports"],
                         title=lang_mapping[lang]["advanced_scatter"],
                            labels={"Distance": lang_mapping[lang]["advanced_scatter_x"],
                                    "Traffic Intensity": lang_mapping[lang]["advanced_scatter_y"]})
st.plotly_chart(scatter_fig, use_container_width=True)

# Advanced Chart: Heatmap for Average Traffic Intensity by Day & Weather
st.subheader(lang_mapping[lang]["advanced_heatmap"])
heatmap_data = filtered_data.groupby(["Day of the Week", "Weather Conditions"])["Traffic Intensity"].mean().reset_index()
heatmap_fig = px.density_heatmap(heatmap_data, x="Day of the Week", y="Weather Conditions",
                                 z="Traffic Intensity", color_continuous_scale="Viridis",
                                 title=lang_mapping[lang]["advanced_heatmap_title"],
                                    labels={"Traffic Intensity": lang_mapping[lang]["advanced_heatmap_z"],
                                            "Day of the Week": lang_mapping[lang]["advanced_heatmap_x"],
                                            "Weather Conditions": lang_mapping[lang]["advanced_heatmap_y"]})
st.plotly_chart(heatmap_fig, use_container_width=True)

################################################################################
#                          FUTURE IMPROVEMENTS & NOTES                         #
################################################################################

st.markdown(f"## {lang_mapping[lang]['future_improvements']}")
st.info(lang_mapping[lang]["future_improvements_note"])

################################################################################
#                          FOOTER & ADDITIONAL INFORMATION                     #
################################################################################

st.markdown(f"### {lang_mapping[lang]['footer']}")

################################################################################
#                              ADDITIONAL UTILITY CODE                         #
################################################################################
# The following code is reserved for future functionality such as exporting data,
# advanced routing analysis, and integration with external APIs.

def export_filtered_data(dataframe, filename="./data/filtered_data.csv"):
    """
    Export the filtered dataframe to a CSV file.
    """
    dataframe.to_csv(filename, index=False)
    st.success(f"Filtered data exported to {filename}")

if st.sidebar.button(lang_mapping[lang]["side_fil_button"]):
    export_filtered_data(filtered_data)

def show_data_summary(df):
    """
    Display a detailed summary of the dataset.
    """
    st.markdown("### Data Summary")
    st.write("Number of records:", df.shape[0])
    st.write("Columns:", list(df.columns))
    st.write(df.describe())

if st.sidebar.checkbox(lang_mapping[lang]["data_summary"]):
    show_data_summary(filtered_data)