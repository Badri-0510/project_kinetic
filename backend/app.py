from flask import Flask, request, send_file, jsonify
import os
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests
import torch
from torchvision import transforms
from scipy.optimize import curve_fit
#from google.colab import files
import tempfile


app = Flask(__name__)
CORS(app)

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

# Load Keras model for No-Ball prediction
model_noball = load_model("model_epoch_23.h5")
# At the top, replace model with a distinct name if using two:
model_yolo = YOLO("cork_prev.pt")

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

# Set device only once
device = torch.device( "cpu")
midas.to(device)
midas_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
])



# Telegram bot config
BOT_TOKEN = "8194662324:AAFoKrZ0SoCD4301Nsj95VB82rWaYtw2cIo"
CHAT_ID = "7662172758"



def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    requests.post(url, data=data)

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

@app.route('/predict_noball', methods=['POST'])
def predict_noball():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save and preprocess image
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))  # assuming VGG16 input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalization

    prediction = model_noball.predict(img_array)[0]
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction))

    labels = ["No-Ball", "Legal Ball"]  # Assuming class 0: No-Ball, class 1: Legal
    result = labels[predicted_class]

    if result == "No-Ball":
        send_telegram_message(f"🚨 No-Ball Detected! Confidence: {confidence:.2f}")

    return jsonify({
        "prediction": result,
        "confidence": confidence
    })


@app.route('/detect_ball_trajectory', methods=['POST'])
def detect_ball_trajectory():
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video_file = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, "input_video.mp4")
    video_file.save(video_path)

    # Open video file twice for separate processing
    cap1 = cv2.VideoCapture(video_path)
    cap2 = cv2.VideoCapture(video_path)

    if not cap1.isOpened() or not cap2.isOpened():
        return jsonify({"error": "Could not open video file"}), 500

    frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap1.get(cv2.CAP_PROP_FPS))

    out1 = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    out2 = cv2.VideoWriter("swing_detection.avi", cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

    trajectory = []
    ball_positions = []

    # ===== Pass 1: Ball Trajectory using YOLO + MiDaS =====
    while cap1.isOpened():
        ret, frame = cap1.read()
        if not ret:
            break

        # Use YOLO and MiDaS for detection
        detection_frame = frame.copy()  # This is where detection happens
        overlay = frame.copy()

        results = model_yolo(detection_frame)
        input_img = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
        input_tensor = midas_transform(input_img).unsqueeze(0)

        with torch.no_grad():
            depth_map = midas(input_tensor).squeeze().numpy()

        # Detect ball and calculate trajectory
        for result in results:
            for box in result.boxes:
                conf = float(box.conf.squeeze())
                if conf < 0.2:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                scaled_cx = min(max(int(cx * (256 / frame_width)), 0), 255)
                scaled_cy = min(max(int(cy * (256 / frame_height)), 0), 255)
                depth_value = depth_map[scaled_cy, scaled_cx]

                trajectory.append((cx, cy, depth_value))

        # Directly draw the trajectory on the frame itself
        if len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                x1, y1, d1 = trajectory[i - 1]
                x2, y2, d2 = trajectory[i]
                intensity = int(255 * (1 - min(d2 / np.max(depth_map), 1)))
                beam_color = (intensity, 0, 255 - intensity)

                # Draw the trajectory line directly on the frame (in the original color)
                cv2.line(frame, (x1, y1), (x2, y2), beam_color, 8, cv2.LINE_AA)

        # Final frame with trajectory plotted on the original frame
        out1.write(frame)

    cap1.release()
    out1.release()

   

    return send_file("output.mp4", as_attachment=True)





# ✅ Ball detection helper
def detect_ball_from_model(model, frame):
    
    results = model_yolo(frame)
    if results and results[0].boxes is not None and len(results[0].boxes.xyxy) > 0:
        box = results[0].boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = box
        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)
        return (x_center, y_center)
    return None

# ✅ Route for bowling analysis
@app.route('/analyze_bowling', methods=['POST'])
def analyze_bowling():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    video_file = request.files['video']
    temp_video_path = os.path.join(tempfile.gettempdir(), video_file.filename)
    video_file.save(temp_video_path)

    cap = cv2.VideoCapture(temp_video_path)
    trajectory, detected_frames = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pos = detect_ball_from_model(model, frame)
        if pos:
            trajectory.append(pos)
            detected_frames.append(frame.copy())
    cap.release()

    if len(trajectory) <= 5:
        return jsonify({'error': 'Not enough ball detections for analysis'}), 400

    y_values = [y for (_, y) in trajectory]
    bounce_index = np.argmax(y_values)
    stumps_y = int(np.mean([y for (_, y) in trajectory[:3]]))
    bounce_y = trajectory[bounce_index][1]

    known_pixel_distance = abs(stumps_y - (stumps_y + 60))
    meters_per_pixel = 1.22 / known_pixel_distance
    distance = abs(bounce_y - stumps_y) * meters_per_pixel

    if distance < 2:
        length = 'Over-pitched'
    elif 2 <= distance < 4:
        length = 'Full'
    elif 4 <= distance < 6:
        length = 'Good'
    elif 6 <= distance < 8:
        length = 'Short'
    else:
        length = 'Very Short'

    # Annotate final frame
    final_frame = detected_frames[bounce_index].copy()
    h, w, _ = final_frame.shape
    regions = [
        (1, 2, 'Over-pitched', (255, 0, 255)),
        (2, 3, 'Full', (0, 255, 0)),
        (3, 4, 'Good', (0, 0, 255)),
        (4, 5, 'Short', (255, 255, 0))
    ]

    for start_m, end_m, label, color in regions:
        y1 = stumps_y + int(start_m / meters_per_pixel)
        y2 = stumps_y + int(end_m / meters_per_pixel)
        overlay = final_frame.copy()
        cv2.rectangle(overlay, (0, y1), (w, y2), color, -1)
        alpha = 0.3
        final_frame = cv2.addWeighted(overlay, alpha, final_frame, 1 - alpha, 0)
        cv2.putText(final_frame, label, (10, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    for (x, y) in trajectory:
        cv2.circle(final_frame, (x, y), 4, (0, 255, 255), -1)

    bx, by = trajectory[bounce_index]
    cv2.circle(final_frame, (bx, by), 8, (0, 0, 255), -1)
    cv2.putText(final_frame, "Bounce", (bx + 10, by), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(final_frame, f"Length: {length}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(final_frame, f"Distance: {round(distance, 2)} m", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Save annotated frame
    output_path = os.path.join(tempfile.gettempdir(), 'bowling_analysis.jpg')
    cv2.imwrite(output_path, final_frame)

    return send_file(output_path, mimetype='image/jpeg')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
