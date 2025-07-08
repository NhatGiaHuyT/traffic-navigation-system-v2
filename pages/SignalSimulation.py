import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
import random
import pandas as pd
from datetime import datetime
import os
import altair as alt

from algo import optimize_traffic

# =============================================================================
# CUSTOM CSS FOR ENHANCED VISUAL APPEAL AND READABILITY
# =============================================================================
CUSTOM_CSS = """
<style>
    /* Overall application background with high contrast text */
    .stApp {
        background-size: cover;
        background-position: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f0f2f6;
        color: #333333;
    }
    /* Header styling with shadow for readability */
    .title{
        font-size: 3.5em;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.2em;
        font-weight: 700;
    }
    /* Sidebar text styling */
    .sidebar .sidebar-content {
        font-size: 16px;
        color: #2c3e50;
    }
    /* Card styling for simulation data and signal timings */
    .traffic-card {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 15px;
        margin: 8px 0;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .traffic-card:hover {
        transform: scale(1.03);
        box-shadow: 0px 8px 16px rgba(0,0,0,0.15);
    }
    /* Traffic signal light styling */
    .traffic-light {
        border-radius: 50%;
        width: 25px;
        height: 25px;
        margin: 3px;
        display: inline-block;
        border: 2px solid #333333;
    }
    /* Ensure all text is dark when needed */
    .stMarkdown, .stText, .stTitle, .stHeader, .stSubheader, .stCaption {
        color: #2c3e50 !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============================================================================
# GLOBAL CONSTANTS & SIMULATION SETTINGS
# =============================================================================

MAX_DISTANCE = 100.0          # Maximum distance vehicles travel along a road
ROAD_LENGTH = 100.0           # Length of each road (same as MAX_DISTANCE)
REFRESH_INTERVAL = 1.0        # Refresh interval (seconds) per simulation step
BASE_SPEED = 20.0             # Base vehicle speed (distance units per second)
VEHICLE_LENGTH = 4.0          # Vehicle length (for drawing rectangles)
VEHICLE_WIDTH = 2.0           # Vehicle width (for drawing rectangles)
SIM_DURATION = 300            # Maximum simulation duration in seconds
TIME_SCALE = 1.0              # Simulation time scaling factor

# New constants for multiple lanes and vehicle types
LANES_PER_ROAD = 2

VEHICLE_TYPES = {
    "car": {"length": 4.0, "width": 2.0, "color": "blue"},
    "motorcycle": {"length": 2.0, "width": 1.0, "color": "red"},
    "truck": {"length": 6.0, "width": 3.0, "color": "green"},
    "bus": {"length": 8.0, "width": 3.0, "color": "orange"}
}

# =============================================================================
# SESSION STATE INITIALISATION (if not already set)
# =============================================================================
if "sim_running" not in st.session_state:
    st.session_state.sim_running = False
if "sim_time" not in st.session_state:
    st.session_state.sim_time = 0.0
if "current_road_index" not in st.session_state:
    st.session_state.current_road_index = 0
if "remaining_green_time" not in st.session_state:
    st.session_state.remaining_green_time = 0.0
if "vehicle_positions" not in st.session_state:
    # Each vehicle represented as dict: {"pos": float, "lane": int, "type": str, "orientation": str}
    st.session_state.vehicle_positions = {
        "Road A": [],
        "Road B": [],
        "Road C": [],
        "Road D": []
    }
if "traffic_volumes" not in st.session_state:
    st.session_state.traffic_volumes = [10, 10, 10, 10]
if "timings" not in st.session_state:
    st.session_state.timings = [(15, 4, 6)] * 4

# Override timings with optimized green times if available
def get_current_timings():
    base_yellow = 4
    base_red = 6
    optimized = st.session_state.get("optimized_timings", None)
    if optimized:
        return [
            (optimized["north"], base_yellow, base_red),
            (optimized["south"], base_yellow, base_red),
            (optimized["west"], base_yellow, base_red),
            (optimized["east"], base_yellow, base_red),
        ]
    else:
        return st.session_state.timings
if "weather_condition" not in st.session_state:
    st.session_state.weather_condition = "clear"
if "road_condition" not in st.session_state:
    st.session_state.road_condition = "Normal"

directions = ["Road A", "Road B", "Road C", "Road D"]

# =============================================================================
# APPLICATION HEADER & NAVIGATION
# =============================================================================
st.markdown("<h1 class='title'>🚦 Traffic Signal Simulation & Optimisation 🚦</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='title'>Experience a real-time simulation of traffic signals, vehicle flows, and adaptive signal timings!</h3>", unsafe_allow_html=True)
st.markdown("---")

# =============================================================================
# SIDEBAR: USER INPUTS & SETTINGS
# =============================================================================
with st.sidebar:
    st.header("Simulation Settings")

    with st.expander("Weather Settings", expanded=True):
        weather = st.selectbox("Select Weather Condition", ["clear", "rainy", "foggy"], index=["clear", "rainy", "foggy"].index(st.session_state.weather_condition))
        st.session_state.weather_condition = weather

    with st.expander("Road Conditions", expanded=True):
        road_cond = st.selectbox("Select Road Condition", 
                                 ["Normal", "Road Closure", "Traffic Incident", "Construction Zone", "Emergency Response"], index=["Normal", "Road Closure", "Traffic Incident", "Construction Zone", "Emergency Response"].index(st.session_state.road_condition))
        st.session_state.road_condition = road_cond

    traffic_input = []
    closed_road = None
    affected_road = None

    if road_cond == "Normal":
        for d in directions:
            tv = st.number_input(f"Traffic volume for {d} (veh/min):", min_value=0, value=10, step=1)
            traffic_input.append(tv)
    elif road_cond == "Road Closure":
        closed_road = st.selectbox("Select Road to Close", directions)
        for d in directions:
            if d == closed_road:
                traffic_input.append(0)
            else:
                tv = st.number_input(f"Traffic volume for {d} (veh/min):", min_value=0, value=30, step=1)
                traffic_input.append(tv)
    else:
        affected_road = st.selectbox("Select Affected Road", directions)
        for d in directions:
            tv = st.number_input(f"Traffic volume for {d} (veh/min):", min_value=0, value=30, step=1)
            if d == affected_road:
                if road_cond == "Traffic Incident":
                    tv = int(tv * 1.5)
                elif road_cond == "Construction Zone":
                    tv = int(tv * 1.3)
            traffic_input.append(tv)

    adjusted_volumes = traffic_input
    adjusted_volumes = [v * 1.8 if weather == "rainy" else v * 1.5 if weather == "foggy" else v for v in adjusted_volumes]

    method = st.sidebar.radio("Optimization Method", options=["genetic", "ml"], index=0, help="Select optimization algorithm")
    method = st.sidebar.radio("Optimization Method", options=["genetic", "ml"], index=0, key="opt_method_radio", help="Select optimization algorithm")
    if st.button("Optimize Signal Timings"):
        optimized_timings = optimize_traffic(adjusted_volumes, method=method, verbose=True)
        st.session_state.optimized_timings = optimized_timings
        st.success(f"Signal timings optimized using {method} method!")

    if "optimized_timings" not in st.session_state:
        st.session_state.optimized_timings = {
            "north": 15,
            "south": 15,
            "west": 15,
            "east": 15
        }

    settings = {
        "volumes": adjusted_volumes,
        "weather_condition": weather,
        "road_condition": road_cond,
        "closed_road": closed_road,
        "affected_road": affected_road
    }

    if st.button("Apply Settings"):
        st.session_state.traffic_volumes = settings["volumes"]
        st.session_state.timings = calculate_timings(
            st.session_state.traffic_volumes,
            scenario=settings["road_condition"],
            weather=settings["weather_condition"],
            closed=settings["closed_road"],
            incident=settings["affected_road"],
            priority=settings["affected_road"]
        )
        st.session_state.sim_time = 0.0
        st.session_state.current_road_index = 0
        st.session_state.remaining_green_time = 0.0
        st.session_state.vehicle_positions = {d: [] for d in directions}
        st.session_state.sim_running = False
        st.success("Settings applied. Simulation reset.")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Traffic Data", "Signal Timings", "Animation"])

with tab1:
    st.header("Simulation Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Simulation Time (s)", f"{st.session_state.sim_time:.1f}")
    col2.metric("Active Road", directions[st.session_state.current_road_index])
    col3.metric("Remaining Green Time (s)", f"{st.session_state.remaining_green_time:.1f}")
    total_vehicles = sum(len(st.session_state.vehicle_positions[d]) for d in directions)
    st.metric("Total Vehicles in Simulation", total_vehicles)

with tab2:
    st.header("Traffic Volume and Vehicle Types")
    # Traffic volume bar chart with Altair
    volumes = st.session_state.traffic_volumes
    df_volumes = pd.DataFrame({
        "Road": directions,
        "Volume": volumes
    })
    bar_chart = alt.Chart(df_volumes).mark_bar().encode(
        x='Road',
        y='Volume',
        color=alt.Color('Volume:Q', scale=alt.Scale(domain=[0, 20, 40, 60], range=['green', 'green', 'orange', 'red']), legend=None)
    ).properties(title="Current Traffic Volumes")
    st.altair_chart(bar_chart, use_container_width=True)

    # Vehicle type distribution pie chart
    vehicle_counts = {vtype: 0 for vtype in VEHICLE_TYPES.keys()}
    for road in directions:
        for vehicle in st.session_state.vehicle_positions[road]:
            vehicle_counts[vehicle["type"]] += 1
    df_vehicle_types = pd.DataFrame({
        "Vehicle Type": list(vehicle_counts.keys()),
        "Count": list(vehicle_counts.values())
    })
    pie_chart = alt.Chart(df_vehicle_types).mark_arc(innerRadius=50).encode(
        theta="Count",
        color="Vehicle Type",
        tooltip=["Vehicle Type", "Count"]
    ).properties(title="Vehicle Type Distribution")
    st.altair_chart(pie_chart, use_container_width=True)

    # Vehicle distribution heatmap
    heatmap_data = []
    for road in directions:
        for vehicle in st.session_state.vehicle_positions[road]:
            heatmap_data.append({
                "Road": road,
                "Lane": vehicle["lane"],
                "Position": vehicle["pos"]
            })
    if heatmap_data:
        df_heatmap = pd.DataFrame(heatmap_data)
        heatmap_chart = alt.Chart(df_heatmap).mark_rect().encode(
            x=alt.X('Position:Q', bin=alt.Bin(maxbins=30), title='Position along road'),
            y=alt.Y('Lane:O', title='Lane'),
            color=alt.Color('count()', scale=alt.Scale(scheme='reds'), legend=alt.Legend(title='Vehicle Count')),
            tooltip=['Road', 'Lane', 'count()']
        ).properties(
            title='Vehicle Distribution Heatmap',
            width=700,
            height=200
        ).facet(
            column='Road:N'
        )
        st.altair_chart(heatmap_chart, use_container_width=True)

with tab3:
    st.header("Signal Timings")
    timings = get_current_timings()
    df_timings = pd.DataFrame({
        "Road": directions,
        "Green (s)": [t[0] for t in timings],
        "Yellow (s)": [t[1] for t in timings],
        "Red (s)": [t[2] for t in timings]
    })
    st.dataframe(df_timings.style.highlight_max(axis=0), use_container_width=True)

    # Prepare data for Gantt chart of signal phases
    gantt_data = []
    for idx, road in enumerate(directions):
        green, yellow, red = timings[idx]
        start = 0
        gantt_data.append({"Road": road, "Phase": "Green", "Start": start, "End": start + green})
        start += green
        gantt_data.append({"Road": road, "Phase": "Yellow", "Start": start, "End": start + yellow})
        start += yellow
        gantt_data.append({"Road": road, "Phase": "Red", "Start": start, "End": start + red})

    df_gantt = pd.DataFrame(gantt_data)
    gantt_chart = alt.Chart(df_gantt).mark_bar().encode(
        x=alt.X('Start:Q', title='Time (s)'),
        x2='End:Q',
        y=alt.Y('Road:N', title='Road'),
        color=alt.Color('Phase:N', scale=alt.Scale(domain=['Green', 'Yellow', 'Red'], range=['#66cc66', '#ffcc00', '#ff4d4d'])),
        tooltip=['Road', 'Phase', alt.Tooltip('Start:Q', title='Start (s)'), alt.Tooltip('End:Q', title='End (s)')]
    ).properties(
        title='Signal Phase Timings',
        width=700,
        height=200
    )
    st.altair_chart(gantt_chart, use_container_width=True)

with tab4:
    st.header("Intersection Animation")
    def draw_traffic_lights(ax):
        positions = {
            "Road A": (30, 55),
            "Road B": (55, 30),
            "Road C": (70, 55),
            "Road D": (55, 70)
        }
        light_colors = {
            "red": "#ff4d4d",
            "yellow": "#ffcc00",
            "green": "#66cc66",
            "off": "#cccccc"
        }
        for idx, road in enumerate(directions):
            x, y = positions[road]
            red_state = "off"
            yellow_state = "off"
            green_state = "off"
            if idx == st.session_state.current_road_index:
                if st.session_state.remaining_green_time < 3:
                    yellow_state = "yellow"
                else:
                    green_state = "green"
            else:
                red_state = "red"
            ax.add_patch(patches.Circle((x, y + 8), 3, color=light_colors[red_state], ec="black"))
            ax.add_patch(patches.Circle((x, y + 4), 3, color=light_colors[yellow_state], ec="black"))
            ax.add_patch(patches.Circle((x, y), 3, color=light_colors[green_state], ec="black"))

    def draw_vehicle(ax, pos, offset, orientation, vehicle_type):
        v = VEHICLE_TYPES[vehicle_type]
        L, W = v["length"]*1.5, v["width"]*1.5
        rect = patches.Rectangle((-L/2, -W/2), L, W,
                                 linewidth=1, edgecolor="black", facecolor=v["color"])
        t = plt.matplotlib.transforms.Affine2D()
        if orientation == "vertical":
            t = t.rotate_deg(90)
        x = pos if orientation=="horizontal" else 50 + offset
        y = 50 + offset if orientation=="horizontal" else pos
        t = t.translate(x, y)
        rect.set_transform(t + ax.transData)
        ax.add_patch(rect)
        arrow = patches.FancyArrow(0, 0, L/2, 0,
                                   width=W/4,
                                   length_includes_head=True,
                                   transform=t + ax.transData,
                                   color="black")
        ax.add_patch(arrow)

    def draw_intersection():
        fig, ax = plt.subplots(figsize=(6,6))
        color_active = "#c5f7c5"
        color_inactive = "#dddddd"
        color_A = color_active if st.session_state.current_road_index == 0 else color_inactive
        ax.add_patch(patches.Rectangle((0, 45), 50, 10, color=color_A))
        color_B = color_active if st.session_state.current_road_index == 1 else color_inactive
        ax.add_patch(patches.Rectangle((45, 0), 10, 50, color=color_B))
        color_C = color_active if st.session_state.current_road_index == 2 else color_inactive
        ax.add_patch(patches.Rectangle((50, 45), 50, 10, color=color_C))
        color_D = color_active if st.session_state.current_road_index == 3 else color_inactive
        ax.add_patch(patches.Rectangle((45, 50), 10, 50, color=color_D))
        draw_traffic_lights(ax)
        lane_width = 10
        lane_offsets = {0: -lane_width, 1: 0, 2: lane_width}
        for road in directions:
            for vehicle in st.session_state.vehicle_positions[road]:
                pos = vehicle["pos"]
                offset = lane_offsets[vehicle["lane"]]
                orientation = vehicle["orientation"]
                vtype = vehicle["type"]
                if road == "Road A":
                    draw_vehicle(ax, pos=pos, offset=offset, orientation="horizontal", vehicle_type=vtype)
                elif road == "Road B":
                    draw_vehicle(ax, pos=pos, offset=offset, orientation="vertical", vehicle_type=vtype)
                elif road == "Road C":
                    draw_vehicle(ax, pos=MAX_DISTANCE - pos, offset=offset, orientation="horizontal", vehicle_type=vtype)
                elif road == "Road D":
                    draw_vehicle(ax, pos=MAX_DISTANCE - pos, offset=offset, orientation="vertical", vehicle_type=vtype)
        for lane_num, offset in lane_offsets.items():
            ax.axhline(y=50 + offset + lane_width/2, color='black', linestyle='--', linewidth=1)
            ax.axhline(y=50 + offset - lane_width/2, color='black', linestyle='--', linewidth=1)
            ax.axvline(x=50 + offset + lane_width/2, color='black', linestyle='--', linewidth=1)
            ax.axvline(x=50 + offset - lane_width/2, color='black', linestyle='--', linewidth=1)
        ax.set_xlim(0, MAX_DISTANCE)
        ax.set_ylim(0, MAX_DISTANCE)
        ax.set_aspect('equal')
        ax.axis('off')
        st.pyplot(fig)

    draw_intersection()

# =============================================================================
# SIMULATION CONTROL BUTTONS
# =============================================================================
control_cols = st.columns([1, 1, 1, 2])
with control_cols[0]:
    if st.button("Start Simulation", key="start_sim"):
        st.session_state.sim_running = True
with control_cols[1]:
    if st.button("Stop Simulation", key="stop_sim"):
        st.session_state.sim_running = False
with control_cols[2]:
    if st.button("Step Simulation", key="step_sim"):
        run_simulation_step(REFRESH_INTERVAL)
with control_cols[3]:
    auto_chk = st.checkbox("Continuous Animation", value=st.session_state.sim_running, key="auto_anim")
    st.session_state.sim_running = auto_chk

# =============================================================================
# SIMULATION LOGIC FUNCTIONS
# =============================================================================

def calculate_timings(volumes, scenario, weather, closed=None, incident=None, priority=None):
    base_green = 15
    base_yellow = 4
    base_red = 6
    timings = []
    for idx, volume in enumerate(volumes):
        green = base_green + int(volume / 30)
        yellow = base_yellow + int(volume / 50)
        red = base_red + int(volume / 60)
        if weather == "rainy":
            yellow += 2
        if weather == "foggy":
            red += 2
        if scenario == "Emergency Response":
            if idx == priority:
                green = int(green * 1.5)
            else:
                green = int(green * 0.8)
                red = int(red * 0.7)
        elif scenario == "Road Closure" and idx == closed:
            green, yellow, red = 0, 0, 0
        elif scenario in ["Traffic Incident", "Construction Zone"] and idx == incident:
            factor = 1.5 if scenario == "Traffic Incident" else 1.2
            green = int(green * factor)
        timings.append((green, yellow, red))
    return timings

import numpy as np

def spawn_vehicles(directions, volumes):
    orientation_map = {
        "Road A": "horizontal",
        "Road B": "vertical",
        "Road C": "horizontal",
        "Road D": "vertical"
    }
    for i, road in enumerate(directions):
        arrival_rate = volumes[i] / 60.0
        if np.random.rand() < (arrival_rate * REFRESH_INTERVAL):
            lane = np.random.randint(0, LANES_PER_ROAD)
            vehicle_type = np.random.choice(list(VEHICLE_TYPES.keys()))
            orientation = orientation_map.get(road, "horizontal")
            vehicle = {"pos": 0.0, "lane": lane, "type": vehicle_type, "orientation": orientation}
            st.session_state.vehicle_positions[road].append(vehicle)

def move_vehicles(directions, delta_time):
    acceleration = 5.0
    max_speed = BASE_SPEED
    min_spacing = 5.0
    active_road = directions[st.session_state.current_road_index]
    for road in directions:
        vehicles = st.session_state.vehicle_positions[road]
        if not vehicles:
            continue
        # Convert to numpy arrays for vectorized operations
        positions = np.array([v["pos"] for v in vehicles])
        lanes = np.array([v["lane"] for v in vehicles])
        types = [v["type"] for v in vehicles]
        orientations = [v.get("orientation", "horizontal") for v in vehicles]

        speeds = np.array([st.session_state.get(f"{road}_speed_{lane}", 0.0) for lane in lanes])

        new_positions = np.copy(positions)
        new_speeds = np.copy(speeds)

        # Sort vehicles by position within each lane
        sorted_indices = np.argsort(positions)
        positions_sorted = positions[sorted_indices]
        lanes_sorted = lanes[sorted_indices]
        speeds_sorted = speeds[sorted_indices]

        for idx in range(len(positions_sorted)):
            lane = lanes_sorted[idx]
            speed = speeds_sorted[idx]
            pos = positions_sorted[idx]
            if road == active_road:
                speed = min(speed + acceleration * delta_time, max_speed)
                # Check spacing with vehicle ahead in same lane
                if idx > 0 and lanes_sorted[idx-1] == lane:
                    lead_pos = positions_sorted[idx-1]
                    if lead_pos - pos < min_spacing:
                        speed = 0.0
                pos += speed * delta_time
            else:
                speed = max(speed - acceleration * delta_time, 0.0)
                pos += speed * delta_time
            speeds_sorted[idx] = speed
            positions_sorted[idx] = pos

        # Update speeds and positions back to original order
        for i, sorted_idx in enumerate(sorted_indices):
            new_speeds[sorted_idx] = speeds_sorted[i]
            new_positions[sorted_idx] = positions_sorted[i]

        # Update session state speeds and vehicle positions
        new_vehicles = []
        for i, v in enumerate(vehicles):
            lane = lanes[i]
            speed = new_speeds[i]
            pos = new_positions[i]
            st.session_state[f"{road}_speed_{lane}"] = speed
            if pos <= MAX_DISTANCE:
                new_vehicles.append({
                    "pos": pos,
                    "lane": lane,
                    "type": types[i],
                    "orientation": orientations[i]
                })
        st.session_state.vehicle_positions[road] = new_vehicles

def run_simulation_step(delta_time):
    timings = get_current_timings()
    if not timings:
        return
    curr_index = st.session_state.current_road_index
    if st.session_state.remaining_green_time <= 0:
        st.session_state.remaining_green_time = timings[curr_index][0]
    st.session_state.remaining_green_time -= delta_time

    leave_rate = 1.0
    st.session_state.traffic_volumes[curr_index] = max(
        st.session_state.traffic_volumes[curr_index] - (leave_rate * delta_time), 0
    )

    spawn_vehicles(directions, st.session_state.traffic_volumes)
    move_vehicles(directions, delta_time)

    if st.session_state.remaining_green_time <= 0:
        st.session_state.current_road_index = (curr_index + 1) % len(directions)

    st.session_state.sim_time += delta_time

def simulate_accidents():
    accident_chance = 0.05
    last_accident_time = st.session_state.get("last_accident_time", 0)
    current_time = st.session_state.sim_time
    accident_cooldown = 30  # seconds cooldown between accidents

    if current_time - last_accident_time > accident_cooldown:
        if random.random() < accident_chance:
            active_road = directions[st.session_state.current_road_index]
            st.warning(f"⚠️ Accident occurred on {active_road}!")
            st.session_state.traffic_volumes[st.session_state.current_road_index] = max(
                st.session_state.traffic_volumes[st.session_state.current_road_index] * 0.5, 0
            )
            for lane in range(LANES_PER_ROAD):
                st.session_state[f"{active_road}_speed_{lane}"] = max(
                    st.session_state.get(f"{active_road}_speed_{lane}", 0.0) * 0.3, 0.0
                )
            st.session_state["last_accident_time"] = current_time

def log_simulation_data():
    log_entry = {
        "sim_time": st.session_state.sim_time,
        "active_road": directions[st.session_state.current_road_index],
        "volumes": st.session_state.traffic_volumes.copy()
    }
    print(log_entry)

# =============================================================================
# PLAYBACK AND ANIMATION LOOP
# =============================================================================
animation_placeholder = st.empty()
chart_placeholder = st.empty()
time_series_data = []
time_series_timestamps = []

while st.session_state.sim_running and st.session_state.sim_time < SIM_DURATION:
    start_loop_time = time.time()
    run_simulation_step(REFRESH_INTERVAL)

    # Update animation tab content directly in placeholder
    with tab4:
        fig, ax = plt.subplots(figsize=(6,6))
        color_active = "#c5f7c5"
        color_inactive = "#dddddd"
        color_A = color_active if st.session_state.current_road_index == 0 else color_inactive
        ax.add_patch(patches.Rectangle((0, 45), 50, 10, color=color_A))
        color_B = color_active if st.session_state.current_road_index == 1 else color_inactive
        ax.add_patch(patches.Rectangle((45, 0), 10, 50, color=color_B))
        color_C = color_active if st.session_state.current_road_index == 2 else color_inactive
        ax.add_patch(patches.Rectangle((50, 45), 50, 10, color=color_C))
        color_D = color_active if st.session_state.current_road_index == 3 else color_inactive
        ax.add_patch(patches.Rectangle((45, 50), 10, 50, color=color_D))
        light_colors = {
            "red": "#ff4d4d",
            "yellow": "#ffcc00",
            "green": "#66cc66",
            "off": "#cccccc"
        }
        positions = {
            "Road A": (30, 55),
            "Road B": (55, 30),
            "Road C": (70, 55),
            "Road D": (55, 70)
        }
        for idx, road in enumerate(directions):
            x, y = positions[road]
            red_state = "off"
            yellow_state = "off"
            green_state = "off"
            if idx == st.session_state.current_road_index:
                if st.session_state.remaining_green_time < 3:
                    yellow_state = "yellow"
                else:
                    green_state = "green"
            else:
                red_state = "red"
            ax.add_patch(patches.Circle((x, y + 8), 3, color=light_colors[red_state], ec="black"))
            ax.add_patch(patches.Circle((x, y + 4), 3, color=light_colors[yellow_state], ec="black"))
            ax.add_patch(patches.Circle((x, y), 3, color=light_colors[green_state], ec="black"))
        lane_width = 10
        lane_offsets = {0: -lane_width, 1: 0, 2: lane_width}
        for road in directions:
            for vehicle in st.session_state.vehicle_positions[road]:
                pos = vehicle["pos"]
                offset = lane_offsets[vehicle["lane"]]
                orientation = vehicle["orientation"]
                vtype = vehicle["type"]
                if road == "Road A":
                    draw_vehicle(ax, pos=pos, offset=offset, orientation="horizontal", vehicle_type=vtype)
                elif road == "Road B":
                    draw_vehicle(ax, pos=pos, offset=offset, orientation="vertical", vehicle_type=vtype)
                elif road == "Road C":
                    draw_vehicle(ax, pos=MAX_DISTANCE - pos, offset=offset, orientation="horizontal", vehicle_type=vtype)
                elif road == "Road D":
                    draw_vehicle(ax, pos=MAX_DISTANCE - pos, offset=offset, orientation="vertical", vehicle_type=vtype)
        for lane_num, offset in lane_offsets.items():
            ax.axhline(y=50 + offset + lane_width/2, color='black', linestyle='--', linewidth=1)
            ax.axhline(y=50 + offset - lane_width/2, color='black', linestyle='--', linewidth=1)
            ax.axvline(x=50 + offset + lane_width/2, color='black', linestyle='--', linewidth=1)
            ax.axvline(x=50 + offset - lane_width/2, color='black', linestyle='--', linewidth=1)
        ax.set_xlim(0, MAX_DISTANCE)
        ax.set_ylim(0, MAX_DISTANCE)
        ax.set_aspect('equal')
        ax.axis('off')
        animation_placeholder.pyplot(fig)

    timestamp = datetime.now()
    time_series_timestamps.append(timestamp)
    total_in_roi = sum([len(st.session_state.vehicle_positions[d]) for d in directions])
    time_series_data.append(total_in_roi)
    chart_df = pd.DataFrame({
        "Time": time_series_timestamps,
        "Vehicles": time_series_data
    })
    chart_placeholder.empty()
    with tab2:
        line_chart = alt.Chart(chart_df).mark_line(point=True).encode(
            x=alt.X('Time:T', title='Time'),
            y=alt.Y('Vehicles:Q', title='Number of Vehicles'),
            tooltip=[alt.Tooltip('Time:T', title='Time'), alt.Tooltip('Vehicles:Q', title='Vehicles')],
            color=alt.value('steelblue')
        ).properties(
            title="Real-Time Vehicle Count Across All Roads",
            width=700,
            height=300
        ).interactive()
        chart_placeholder.altair_chart(line_chart, use_container_width=True)

    simulate_accidents()
    log_simulation_data()
    elapsed = time.time() - start_loop_time
    time.sleep(max(REFRESH_INTERVAL - elapsed, 0.1))

if not st.session_state.sim_running:
    with animation_placeholder.container():
        with tab4:
            draw_intersection()

# =============================================================================
# FINAL NOTES & FUTURE ENHANCEMENTS
# =============================================================================
st.markdown("---")
st.markdown("**Tip:** Adjust settings in the sidebar and click **Apply Settings** to reset the simulation.")
st.markdown("**Note:** With Continuous Animation enabled, the simulation updates every second.")
st.markdown("Future enhancements may include advanced accident simulation, dynamic ROI drawing, and live data integration.")
