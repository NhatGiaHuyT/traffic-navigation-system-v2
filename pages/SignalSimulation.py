import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
import time
import random
import pandas as pd
from datetime import datetime
import os
import altair as alt

# =============================================================================
# CUSTOM CSS FOR ENHANCED VISUAL APPEAL AND READABILITY
# =============================================================================
CUSTOM_CSS = """
<style>
    /* Overall application background with high contrast text */
    .stApp {
        background-size: cover;
        background-position: center;
    }
    /* Header styling with shadow for readability */
    .title h1 {
        font-size: 3.5em;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.2em;
        text-shadow: 3px 3px 8px #000000;
    }
    /* Sidebar text styling */
    .sidebar .sidebar-content {
        font-size: 16px;
        color: #333333;
    }
    /* Card styling for simulation data and signal timings */
    .traffic-card {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 15px;
        margin: 8px 0;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .traffic-card:hover {
        transform: scale(1.03);
        box-shadow: 0px 8px 16px rgba(0,0,0,0.3);
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
        color: #000000 !important;
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
LANES_PER_ROAD = 3

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
if "weather_condition" not in st.session_state:
    st.session_state.weather_condition = "clear"
if "road_condition" not in st.session_state:
    st.session_state.road_condition = "Normal"

# =============================================================================
# APPLICATION HEADER & NAVIGATION
# =============================================================================
st.sidebar.markdown("## Navigation")
if st.sidebar.button("Home"):
    st.session_state.page = "home"

st.markdown("<div class='title'><h1>🚦 Traffic Signal Simulation & Optimisation 🚦</h1></div>", unsafe_allow_html=True)
st.markdown("### Experience a real-time simulation of traffic signals, vehicle flows, and adaptive signal timings!")
st.markdown("---")

# =============================================================================
# HELPER FUNCTIONS: SIMULATION LOGIC & RENDERING
# =============================================================================

def display_header_image():
    header_url = "https://images.unsplash.com/photo-1516707570268-975c0a813ce0?fm=jpg&q=60&w=1000"
    st.image(header_url, use_column_width=True)

def adjust_for_weather(volumes, condition):
    adjusted = volumes.copy()
    if condition == "rainy":
        adjusted = [v * 1.8 for v in adjusted]
    elif condition == "foggy":
        adjusted = [v * 1.5 for v in adjusted]
    return adjusted

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

import matplotlib.image as mpimg
import os

def draw_vehicle(ax, pos, offset, orientation, vehicle_type):
    """
    Draw a single vehicle at:
      - along‐axis coordinate = pos
      - cross‐axis coordinate = 50 + offset*lane_width
    with correct rotation and an arrow that transforms with it.
    """
    v = VEHICLE_TYPES[vehicle_type]
    L, W = v["length"]*1.5, v["width"]*1.5
    # rectangle centered at origin
    rect = patches.Rectangle((-L/2, -W/2), L, W,
                             linewidth=1, edgecolor="black", facecolor=v["color"])
    # build transform: rotate first, then translate
    t = plt.matplotlib.transforms.Affine2D()
    if orientation == "vertical":
        t = t.rotate_deg(90)
    # translate to global (x,y)
    x = pos if orientation=="horizontal" else 50 + offset
    y = 50 + offset if orientation=="horizontal" else pos
    t = t.translate(x, y)
    rect.set_transform(t + ax.transData)
    ax.add_patch(rect)

    # arrow drawn as a patch so it picks up the same transform
    arrow = patches.FancyArrow(0, 0, L/2, 0,
                               width=W/4,
                               length_includes_head=True,
                               transform=t + ax.transData,
                               color="black")
    ax.add_patch(arrow)


def spawn_vehicles(directions, volumes):
    for i, road in enumerate(directions):
        arrival_rate = volumes[i] / 60.0
        if random.random() < (arrival_rate * REFRESH_INTERVAL):
            # Randomly assign lane, vehicle type, and orientation based on road
            lane = random.randint(0, LANES_PER_ROAD - 1)
            vehicle_type = random.choice(list(VEHICLE_TYPES.keys()))
            orientation_map = {
                "Road A": "horizontal",
                "Road B": "vertical",
                "Road C": "horizontal",
                "Road D": "vertical"
            }
            orientation = orientation_map.get(road, "horizontal")
            vehicle = {"pos": 0.0, "lane": lane, "type": vehicle_type, "orientation": orientation}
            st.session_state.vehicle_positions[road].append(vehicle)

def move_vehicles(directions, delta_time):
    """
    Update vehicle positions with acceleration and deceleration simulation.
    Vehicles on the active road move forward with acceleration.
    Vehicles on other roads slow down or stop.
    """
    acceleration = 5.0  # units per second squared
    max_speed = BASE_SPEED
    min_spacing = 5.0  # minimum spacing between vehicles in the same lane
    active_road = directions[st.session_state.current_road_index]
    for road in directions:
        new_vehicles = []
        # Sort vehicles by position to check spacing
        vehicles_sorted = sorted(st.session_state.vehicle_positions[road], key=lambda v: v["pos"])
        for i, vehicle in enumerate(vehicles_sorted):
            pos = vehicle["pos"]
            lane = vehicle["lane"]
            v_type = vehicle["type"]
            orientation = vehicle.get("orientation", "horizontal")
            # Simulate speed with acceleration/deceleration
            speed = st.session_state.get(f"{road}_speed_{lane}", 0.0)
            if road == active_road:
                speed = min(speed + acceleration * delta_time, max_speed)
                # Check spacing with vehicle ahead in same lane
                if i > 0 and vehicles_sorted[i-1]["lane"] == lane:
                    lead_pos = vehicles_sorted[i-1]["pos"]
                    if lead_pos - pos < min_spacing:
                        speed = 0.0  # stop to avoid collision
                pos += speed * delta_time
            else:
                speed = max(speed - acceleration * delta_time, 0.0)
                pos += speed * delta_time
            st.session_state[f"{road}_speed_{lane}"] = speed
            if pos <= MAX_DISTANCE:
                new_vehicles.append({"pos": pos, "lane": lane, "type": v_type, "orientation": orientation})
        st.session_state.vehicle_positions[road] = new_vehicles

def run_simulation_step(delta_time):
    """
    Perform a single simulation step:
    - Update signal timings and remaining green time.
    - Update vehicle positions based on active road.
    - Spawn new vehicles based on traffic volumes.
    - Advance simulation time.
    """
    if not st.session_state.timings:
        return
    curr_index = st.session_state.current_road_index
    if st.session_state.remaining_green_time <= 0:
        st.session_state.remaining_green_time = st.session_state.timings[curr_index][0]
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

def display_traffic_chart(volumes, directions):
    colors = ['green' if v < 20 else 'orange' if v < 40 else 'red' for v in volumes]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(directions, volumes, color=colors)
    ax.set_title('Current Traffic Volumes', fontsize=18, fontweight='bold')
    ax.set_xlabel('Road', fontsize=14)
    ax.set_ylabel('Volume (veh/min)', fontsize=14)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

def display_result_cards(volumes, timings, directions, active_index, closed=None):
    sorted_data = sorted(
        zip(directions, volumes, timings, range(len(directions))),
        key=lambda x: x[1],
        reverse=True
    )
    for direction, volume, timing, idx in sorted_data:
        if closed is not None and direction == closed:
            continue
        card_class = "traffic-card"
        if idx == active_index:
            card_class += " active-road"
        green, yellow, red = timing
        card_html = f"""
        <div class="{card_class}">
            <h2 style="text-align:center;">{direction}</h2>
            <p style="text-align:center;">Volume: <strong>{int(volume)}</strong> veh/min</p>
            <p style="text-align:center;">Green: <strong>{green:.1f}s</strong></p>
            <p style="text-align:center;">Yellow: <strong>{yellow:.1f}s</strong></p>
            <p style="text-align:center;">Red: <strong>{red:.1f}s</strong></p>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

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
    
    # Increase lane width and spacing for better clarity
    lane_width = 10  # increased from 6 for wider lanes
    lane_offsets = {
        0: -lane_width,
        1: 0,
        2: lane_width
    }

    # Add lane boundary lines for visual clarity
    for lane_num, offset in lane_offsets.items():
        if lane_num == 0:
            ax.axhline(y=50 + offset + lane_width/2, color='black', linestyle='--', linewidth=1)
            ax.axhline(y=50 + offset - lane_width/2, color='black', linestyle='--', linewidth=1)
        elif lane_num == 1:
            ax.axhline(y=50 + offset + lane_width/2, color='black', linestyle='--', linewidth=1)
            ax.axhline(y=50 + offset - lane_width/2, color='black', linestyle='--', linewidth=1)
        elif lane_num == 2:
            ax.axhline(y=50 + offset + lane_width/2, color='black', linestyle='--', linewidth=1)
            ax.axhline(y=50 + offset - lane_width/2, color='black', linestyle='--', linewidth=1)
    # Ensure vehicles stay within lane boundaries
    def constrain_vehicle_position(pos, lane_offset, orientation):
        lane_width = 10
        upper_bound = lane_offset + lane_width / 2
        lower_bound = lane_offset - lane_width / 2
        if orientation == "horizontal":
            # Constrain y position within lane boundaries
            return max(min(pos, upper_bound), lower_bound)
        else:
            # Constrain x position within lane boundaries
            return max(min(pos, upper_bound), lower_bound)

    lane_width = 10
    lane_offsets = {0: -lane_width, 1: 0, 2: +lane_width}

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

    # Draw lane boundary lines for both horizontal and vertical roads
    for lane_num, offset in lane_offsets.items():
        # Horizontal roads: Road A and Road C
        ax.axhline(y=50 + offset + lane_width/2, color='black', linestyle='--', linewidth=1)
        ax.axhline(y=50 + offset - lane_width/2, color='black', linestyle='--', linewidth=1)
        # Vertical roads: Road B and Road D
        ax.axvline(x=50 + offset + lane_width/2, color='black', linestyle='--', linewidth=1)
        ax.axvline(x=50 + offset - lane_width/2, color='black', linestyle='--', linewidth=1)

    ax.set_xlim(0, MAX_DISTANCE)
    ax.set_ylim(0, MAX_DISTANCE)
    ax.set_aspect('equal')
    ax.axis('off')
    st.pyplot(fig)

def update_animation():
    """
    Update all simulation outputs in a single container.
    """
    st.markdown(f"### Simulation Time: {st.session_state.sim_time:.1f}s")
    st.markdown(f"**Active Road (Green):** {directions[st.session_state.current_road_index]}")
    st.markdown(f"**Remaining Green Time:** {st.session_state.remaining_green_time:.1f}s")
    st.markdown("<br>", unsafe_allow_html=True)
    display_traffic_chart(st.session_state.traffic_volumes, directions)
    display_result_cards(st.session_state.traffic_volumes,
                         st.session_state.timings,
                         directions,
                         active_index=st.session_state.current_road_index)
    st.markdown("## Intersection Animation")
    draw_intersection()

def reset_simulation(settings):
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

def simulate_accidents():
    """
    Simulate accidents dynamically with impact on traffic flow.
    When an accident occurs on the active road, reduce traffic volume and slow vehicles.
    Display warning in UI.
    """
    accident_chance = 0.05
    if random.random() < accident_chance:
        active_road = directions[st.session_state.current_road_index]
        st.warning(f"⚠️ Accident occurred on {active_road}!")
        # Reduce traffic volume on active road
        st.session_state.traffic_volumes[st.session_state.current_road_index] = max(
            st.session_state.traffic_volumes[st.session_state.current_road_index] * 0.5, 0
        )
        # Slow vehicles on active road
        for lane in range(LANES_PER_ROAD):
            st.session_state[f"{active_road}_speed_{lane}"] = max(
                st.session_state.get(f"{active_road}_speed_{lane}", 0.0) * 0.3, 0.0
            )

def log_simulation_data():
    log_entry = {
        "sim_time": st.session_state.sim_time,
        "active_road": directions[st.session_state.current_road_index],
        "volumes": st.session_state.traffic_volumes.copy()
    }
    print(log_entry)

def log_data_to_csv(log_df, csv_file="traffic_log.csv"):
    if not os.path.exists(csv_file):
        log_df.to_csv(csv_file, index=False)
    else:
        log_df.to_csv(csv_file, mode='a', header=False, index=False)

# =============================================================================
# SIDEBAR: USER INPUTS & SETTINGS
# =============================================================================
st.sidebar.header("Simulation Settings")

# Group sidebar inputs into expandable sections for better UX
with st.sidebar.expander("Weather Settings", expanded=True):
    weather = st.selectbox("Select Weather Condition", ["clear", "rainy", "foggy"])
    st.session_state.weather_condition = weather

with st.sidebar.expander("Road Conditions", expanded=True):
    road_cond = st.selectbox("Select Road Condition", 
                             ["Normal", "Road Closure", "Traffic Incident", "Construction Zone", "Emergency Response"])
    st.session_state.road_condition = road_cond

directions = ["Road A", "Road B", "Road C", "Road D"]

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

adjusted_volumes = adjust_for_weather(traffic_input, weather)

settings = {
    "volumes": adjusted_volumes,
    "weather_condition": weather,
    "road_condition": road_cond,
    "closed_road": closed_road,
    "affected_road": affected_road
}

if st.button("Apply Settings"):
    reset_simulation(settings)
    st.success("Settings applied. Simulation reset.")

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

# Add clearer status indicators
st.markdown(f"### Simulation Status: {'Running' if st.session_state.sim_running else 'Stopped'}")
st.markdown(f"### Simulation Time: {st.session_state.sim_time:.1f} seconds")
st.markdown(f"### Active Road: {directions[st.session_state.current_road_index]}")
st.markdown(f"### Remaining Green Time: {st.session_state.remaining_green_time:.1f} seconds")

# =============================================================================
# ADDITIONAL REAL-TIME CHARTING (TIME-SERIES)
# =============================================================================
chart_placeholder = st.empty()
time_series_data = []
time_series_timestamps = []

# =============================================================================
# PLAYBACK AND ANIMATION LOOP
# =============================================================================
animation_placeholder = st.empty()
prev_time = time.time()
while st.session_state.sim_running and st.session_state.sim_time < SIM_DURATION:
    start_loop_time = time.time()
    run_simulation_step(REFRESH_INTERVAL)
    
    # Update simulation outputs in a single container to avoid appending new frames
    with animation_placeholder.container():
        update_animation()
    
    # Update real-time chart data in its own container
    timestamp = datetime.now()
    time_series_timestamps.append(timestamp)
    total_in_roi = sum([len(st.session_state.vehicle_positions[d]) for d in directions])
    time_series_data.append(total_in_roi)
    chart_df = pd.DataFrame({
        "Time": time_series_timestamps,
        "Vehicles": time_series_data
    })
    with chart_placeholder.container():
        line_chart = alt.Chart(chart_df).mark_line(point=True).encode(
            x='Time:T',
            y='Vehicles:Q'
        ).properties(title="Real-Time Vehicle Count Across All Roads")
        st.altair_chart(line_chart, use_container_width=True)
    
    simulate_accidents()
    log_simulation_data()
    elapsed = time.time() - start_loop_time
    time.sleep(max(REFRESH_INTERVAL - elapsed, 0.1))
    
# If simulation is stopped, update animation once in the placeholder
if not st.session_state.sim_running:
    with animation_placeholder.container():
        update_animation()

# =============================================================================
# FINAL NOTES & FUTURE ENHANCEMENTS
# =============================================================================
st.markdown("---")
st.markdown("**Tip:** Adjust settings in the sidebar and click **Apply Settings** to reset the simulation.")
st.markdown("**Note:** With Continuous Animation enabled, the simulation updates every second.")
st.markdown("Future enhancements may include advanced accident simulation, dynamic ROI drawing, and live data integration.")

# End of code
