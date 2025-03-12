from flask import Flask, request, send_file, jsonify
import os
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

app = Flask(__name__)

# Create upload directory
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load YOLO model for ball detection
model = YOLO("tennis_ball.pt")  # Ensure 'tennis_ball.pt' is in the same folder

# Initialize MediaPipe Pose for body tracking
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to calculate elbow angle
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, "uploaded.mp4")
    video_file.save(video_path)

    # Process video
    output_video_path = process_video(video_path)

    return send_file(output_video_path, as_attachment=True)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Cannot open video file"}), 500

    # Output settings
    output_path = os.path.join(PROCESSED_FOLDER, "output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    # Tracking variables
    prev_ball_position = None
    prev_wrist_position = None
    prev_elbow_angle = None
    shoulder_level_angle = None
    elbow_angle_at_release = None
    release_detected = False
    chucking_flag = None
    wrist = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape

        # Ball Detection using YOLO
        results = model(frame)
        ball_position = None
        ball_velocity = None
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    ball_position = ((x1 + x2) // 2, (y1 + y2) // 2)
                    cv2.circle(frame, ball_position, 7, (0, 0, 255), -1)
                    break

        if prev_ball_position and ball_position:
            ball_velocity = ball_position[1] - prev_ball_position[1]

        # Pose Detection using MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            def to_pixel(landmark): return int(landmark.x * w), int(landmark.y * h)

            shoulder = to_pixel(lm[12])
            elbow = to_pixel(lm[14])
            wrist = to_pixel(lm[16])

            cv2.circle(frame, shoulder, 8, (255, 0, 0), -1)
            cv2.circle(frame, elbow, 8, (0, 255, 0), -1)
            cv2.circle(frame, wrist, 8, (0, 0, 255), -1)
            cv2.line(frame, shoulder, elbow, (255, 255, 255), 3)
            cv2.line(frame, elbow, wrist, (255, 255, 255), 3)

            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            if prev_elbow_angle is not None:
                if abs(elbow_angle - prev_elbow_angle) > 10:
                    elbow_angle = prev_elbow_angle + np.sign(elbow_angle - prev_elbow_angle) * 10
            prev_elbow_angle = elbow_angle

            cv2.putText(frame, f"Elbow Angle: {elbow_angle:.2f}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if shoulder_level_angle is None and 165 <= elbow_angle <= 180:
                shoulder_level_angle = elbow_angle

            # Ball Release Detection
            if prev_ball_position and ball_position and prev_wrist_position:
                ball_wrist_distance = np.linalg.norm(np.array(ball_position) - np.array(wrist))
                prev_ball_wrist_distance = np.linalg.norm(np.array(prev_ball_position) - np.array(prev_wrist_position))

                condition1 = ball_wrist_distance > prev_ball_wrist_distance + 5
                condition2 = ball_velocity is not None and ball_velocity > 0

                if condition1 or condition2:
                    release_detected = True
                    if elbow_angle_at_release is None:
                        elbow_angle_at_release = elbow_angle

                    cv2.putText(frame, "Ball Released!", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    # Determine Chucking or Legal
                    if shoulder_level_angle is not None and elbow_angle_at_release is not None:
                        angle_difference = shoulder_level_angle - elbow_angle_at_release
                        chucking_flag = "Chucking" if angle_difference > 15 else "Legal"

            if chucking_flag:
                color = (0, 0, 255) if chucking_flag == "Chucking" else (0, 255, 0)
                cv2.putText(frame, f"{chucking_flag}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        prev_ball_position = ball_position
        prev_wrist_position = wrist

        out.write(frame)

    cap.release()
    out.release()
    return output_path

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
