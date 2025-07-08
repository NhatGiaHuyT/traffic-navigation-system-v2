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
                class_name = class_names[int(cls)]
                detections.append([[x1, y1, x2 - x1, y2 - y1], float(conf), int(cls), class_name])
        tracks = tracker.update_tracks(detections, frame=frame)

        # Manually assign det_class_name to each track by matching with detections using IoU
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_bbox = track.to_ltrb()
            best_match = None
            best_iou = 0
            for det in detections:
                det_bbox = det[0]
                iou = calculate_iou(track_bbox, det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = det
            if best_match and best_iou > 0.2:
                track.det_class_name = best_match[3]
            else:
                track.det_class_name = 'Unknown'

        # Collect speed and direction data for visualization
        speed_values = []
        direction_counts = {"Up": 0, "Down": 0, "Left": 0, "Right": 0, "Unknown": 0}
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            # Calculate speed
            if hasattr(track, 'prev_position'):
                prev_position = track.prev_position
                distance_pixels = np.linalg.norm(np.array([x1, y1]) - np.array(prev_position))
                speed_mps = distance_pixels * SCALE_FACTOR
                speed_values.append(speed_mps)
                track.prev_position = [x1, y1]
            else:
                speed_values.append(0.0)
                track.prev_position = [x1, y1]
            # Calculate direction
            direction = detect_direction(track, x1, y1)
            if direction in direction_counts:
                direction_counts[direction] += 1
            else:
                direction_counts["Unknown"] += 1
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
                "speed_values": speed_values,
                "direction_counts": direction_counts,
            }
        time.sleep(0.1)
    cap.release()
