import cv2
import mediapipe as mp
import numpy as np
import os
import math

# --- Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

# --- Configuration ---
DATA_DIR = 'roitraining_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

asl_signs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
             'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

keyframe_map = {
    'J': ['1. Start (Back-of-Hand I)', '2. Mid-Scoop (Turning)', '3. End (Back-of-Hand I)'],
    'Z': ['1. Start (Top-Left)', '2. Corner 1 (Top-Right)', '3. Corner 2 (Bottom-Left)', '4. End (Bottom-Right)']
}

# State variables
current_sign_index = 0
keyframe_index = 0
recording = False
frame_count = 0

# --- Helper Functions ---
def get_hand_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        return results.multi_hand_landmarks[0]
    return None

def extract_hand_roi(frame, hand_landmarks):
    if not hand_landmarks:
        return None, None
    h, w = frame.shape[:2]
    x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    padding_x = int((x_max - x_min) * 0.2)
    padding_y = int((y_max - y_min) * 0.2)
    x_min = max(0, x_min - padding_x)
    x_max = min(w, x_max + padding_x)
    y_min = max(0, y_min - padding_y)
    y_max = min(h, y_max + padding_y)
    roi = frame[y_min:y_max, x_min:x_max]
    if roi.size > 0:
        return roi, (x_min, y_min, x_max, y_max)
    return None, None

def check_wrist_slope(hand_landmarks, image_shape):
    """Calculate the slope between wrist (landmark 0) and pinky MCP (landmark 17)."""
    if not hand_landmarks:
        return None
    h, w = image_shape[:2]
    wrist = (int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h))
    pinky_mcp = (int(hand_landmarks.landmark[17].x * w), int(hand_landmarks.landmark[17].y * h))
    dx = pinky_mcp[0] - wrist[0]
    dy = pinky_mcp[1] - wrist[1]
    if dx == 0:
        slope = float('inf')
    else:
        slope = dy / dx
    return slope, wrist, pinky_mcp

def calculate_slope_and_adjust(hand_landmarks, image_shape, target_size=(200, 200)):
    if not hand_landmarks:
        return None
    h, w = image_shape[:2]
    target_h, target_w = target_size
    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    ring_mcp = landmarks[13]
    pinky_mcp = landmarks[17]
    x_range_min = min(index_mcp[0], middle_mcp[0], ring_mcp[0], pinky_mcp[0])
    x_range_max = max(index_mcp[0], middle_mcp[0], ring_mcp[0], pinky_mcp[0])
    x_range = x_range_max - x_range_min
    x_alignment_threshold = w * 0.1
    is_vertical = x_range_min <= wrist[0] <= x_range_max
    slope_info = check_wrist_slope(hand_landmarks, image_shape)
    slope = None
    if slope_info:
        slope, _, _ = slope_info
    is_horizontal = (wrist[0] < x_range_min or wrist[0] > x_range_max) or (slope is not None and -0.5 <= slope <= 0.5)
    if is_vertical and not is_horizontal:
        fixed_wrist = (target_w // 2, target_h - 50)
        mcp_center_x = (index_mcp[0] + middle_mcp[0] + ring_mcp[0] + pinky_mcp[0]) // 4
        mcp_center_y = (index_mcp[1] + middle_mcp[1] + ring_mcp[1] + pinky_mcp[1]) // 4
        mcp_center = (mcp_center_x, mcp_center_y)
        dx = mcp_center[0] - wrist[0]
        dy = mcp_center[1] - wrist[1]
        if dx != 0 or dy != 0:
            current_angle = math.atan2(dy, dx)
            target_angle = -math.pi / 2
            rotation_angle = target_angle - current_angle
        else:
            rotation_angle = 0
        rotated_landmarks = []
        cos_theta = math.cos(rotation_angle)
        sin_theta = math.sin(rotation_angle)
        for x, y in landmarks:
            rel_x = x - wrist[0]
            rel_y = y - wrist[1]
            new_x = rel_x * cos_theta - rel_y * sin_theta + fixed_wrist[0]
            new_y = rel_x * sin_theta + rel_y * cos_theta + fixed_wrist[1]
            rotated_landmarks.append((int(new_x), int(new_y)))
        transformed_landmarks = rotated_landmarks
    elif is_horizontal:
        if slope is not None and -0.5 <= slope <= 0.5:
            dx = pinky_mcp[0] - wrist[0]
            dy = pinky_mcp[1] - wrist[1]
            if dx != 0 or dy != 0:
                current_angle = math.atan2(dy, dx)
                target_angle = 0
                rotation_angle = target_angle - current_angle
            else:
                rotation_angle = 0
            rotated_landmarks = []
            cos_theta = math.cos(rotation_angle)
            sin_theta = math.sin(rotation_angle)
            for x, y in landmarks:
                rel_x = x - wrist[0]
                rel_y = y - wrist[1]
                new_x = rel_x * cos_theta - rel_y * sin_theta + wrist[0]
                new_y = rel_x * sin_theta + rel_y * cos_theta + wrist[1]
                rotated_landmarks.append((int(new_x), int(new_y)))
            transformed_landmarks = rotated_landmarks
        elif slope is not None and slope > 0.5 and x_range >= x_alignment_threshold:
            fixed_wrist = (target_w // 2, target_h - 50)
            mcp_center_x = (index_mcp[0] + middle_mcp[0] + ring_mcp[0] + pinky_mcp[0]) // 4
            mcp_center_y = (index_mcp[1] + middle_mcp[1] + ring_mcp[1] + pinky_mcp[1]) // 4
            mcp_center = (mcp_center_x, mcp_center_y)
            dx = mcp_center[0] - wrist[0]
            dy = mcp_center[1] - wrist[1]
            if dx != 0 or dy != 0:
                current_angle = math.atan2(dy, dx)
                target_angle = -math.pi / 2
                rotation_angle = target_angle - current_angle
            else:
                rotation_angle = 0
            rotated_landmarks = []
            cos_theta = math.cos(rotation_angle)
            sin_theta = math.sin(rotation_angle)
            for x, y in landmarks:
                rel_x = x - wrist[0]
                rel_y = y - wrist[1]
                new_x = rel_x * cos_theta - rel_y * sin_theta + fixed_wrist[0]
                new_y = rel_x * sin_theta + rel_y * cos_theta + fixed_wrist[1]
                rotated_landmarks.append((int(new_x), int(new_y)))
            transformed_landmarks = rotated_landmarks
        else:
            transformed_landmarks = landmarks
    else:
        transformed_landmarks = landmarks
    x_coords = [pt[0] for pt in transformed_landmarks]
    y_coords = [pt[1] for pt in transformed_landmarks]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    padding = 20
    scale = min((target_w - 2 * padding) / (x_max - x_min) if x_max > x_min else 1,
                (target_h - 2 * padding) / (y_max - y_min) if y_max > y_min else 1)
    offset_x = (target_w - ((x_max - x_min) * scale)) // 2
    offset_y = (target_h - ((y_max - y_min) * scale)) // 2
    final_landmarks = []
    for x, y in transformed_landmarks:
        new_x = int((x - x_min) * scale + offset_x)
        new_y = int((y - y_min) * scale + offset_y)
        final_landmarks.append((new_x, new_y))
    landmark_image = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    connections = mp_hands.HAND_CONNECTIONS
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(final_landmarks) and end_idx < len(final_landmarks):
            cv2.line(landmark_image, final_landmarks[start_idx], final_landmarks[end_idx], (0, 0, 0), 1)
    for point in final_landmarks:
        cv2.circle(landmark_image, point, 3, (0, 0, 0), 1)
    if len(final_landmarks) >= 18:
        cv2.line(landmark_image, final_landmarks[0], final_landmarks[17], (255, 0, 0), 2)
    return landmark_image

def get_existing_count(sign_name):
    sign_dir = os.path.join(DATA_DIR, sign_name)
    if os.path.exists(sign_dir):
        return len([f for f in os.listdir(sign_dir) if f.endswith('.jpg')])
    return 0

# --- Main Loop ---
print("ASL Data Collection - Full Keyframe & Multi-Window Method")
print("=" * 60)
print("Controls:")
print("  'n'/'p' - Next/Previous Letter | 'a'/'d' - Next/Previous Keyframe (J,Z)")
print("  'r' - Start/Stop Recording     | 'q' - Quit")
print("=" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)

    hand_roi, landmark_image_clean, bbox = None, None, None
    white_image_landmarks = np.ones((frame.shape[0], frame.shape[1], 3), dtype=np.uint8) * 255
    hand_landmarks = get_hand_landmarks(frame)
    current_sign = asl_signs[current_sign_index]

    if hand_landmarks:
        hand_roi, bbox = extract_hand_roi(frame, hand_landmarks)
        landmark_image_clean = calculate_slope_and_adjust(hand_landmarks, frame.shape)
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(45, 45, 255), thickness=2))
        mp_drawing.draw_landmarks(white_image_landmarks, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2))
        if bbox:
            color = (0, 255, 0) if not recording else (0, 0, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        slope_info = check_wrist_slope(hand_landmarks, frame.shape)
        if slope_info:
            _, wrist, pinky_mcp = slope_info
            cv2.line(frame, wrist, pinky_mcp, (255, 0, 0), 2)

    existing_count = get_existing_count(current_sign)
    cv2.putText(frame, f"Letter: {current_sign}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if current_sign in keyframe_map:
        num_keyframes = len(keyframe_map[current_sign])
        pose_description = keyframe_map[current_sign][keyframe_index]
        cv2.putText(frame, f"Pose ({keyframe_index + 1}/{num_keyframes}): {pose_description}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Pose: Hold sign clearly.", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Saved: {existing_count} | Session: {frame_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, "n/p:Ltr | a/d:Pose | r:Rec | q:Quit", (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if recording:
        cv2.putText(frame, "REC", (frame.shape[1] - 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if landmark_image_clean is not None:
            sign_dir = os.path.join(DATA_DIR, current_sign)
            if not os.path.exists(sign_dir):
                os.makedirs(sign_dir)
            filename = f'{existing_count + frame_count + 1:04d}.jpg'
            cv2.imwrite(os.path.join(sign_dir, filename), landmark_image_clean)
            frame_count += 1
        else:
            cv2.putText(frame, "NO HAND", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.imshow('Camera Feed', frame)
    if hand_roi is not None:
        cv2.imshow('Hand ROI', cv2.resize(hand_roi, (250, 250)))
    else:
        cv2.imshow('Hand ROI', np.zeros((250, 250, 3), dtype=np.uint8))
    if landmark_image_clean is not None:
        display_image = cv2.resize(landmark_image_clean, (400, 400), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Training Data', display_image)
    else:
        cv2.imshow('Training Data', np.ones((400, 400, 3), dtype=np.uint8) * 255)
    cv2.imshow('Full Frame Landmarks', white_image_landmarks)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        recording = not recording
        if recording:
            frame_count = 0
            print(f"Started recording for '{current_sign}' - Pose {keyframe_index + 1 if current_sign in keyframe_map else ''}")
        else:
            print(f"Stopped. Captured {frame_count} frames for '{current_sign}'. Total: {get_existing_count(current_sign)}")
    elif key == ord('n'):
        current_sign_index = (current_sign_index + 1) % len(asl_signs)
        keyframe_index = 0
        recording = False
        frame_count = 0
        print(f"Switched to sign: {asl_signs[current_sign_index]}")
    elif key == ord('p'):
        current_sign_index = (current_sign_index - 1 + len(asl_signs)) % len(asl_signs)
        keyframe_index = 0
        recording = False
        frame_count = 0
        print(f"Switched to sign: {asl_signs[current_sign_index]}")
    elif key == ord('d'):
        if current_sign in keyframe_map:
            keyframe_index = (keyframe_index + 1) % len(keyframe_map[current_sign])
    elif key == ord('a'):
        if current_sign in keyframe_map:
            keyframe_index = (keyframe_index - 1 + len(keyframe_map[current_sign])) % len(keyframe_map[current_sign])

cap.release()
cv2.destroyAllWindows()
