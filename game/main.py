import os
import cv2
import mediapipe as mp
import torch
import numpy as np
import random
from utils.classifier import PoseClassifier

def extract_landmarks(pose_results, hand_results):
    landmarks = []
    upper_pose_indices = [0, 11, 12, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    if pose_results.pose_landmarks:
        for idx in upper_pose_indices:
            lm = pose_results.pose_landmarks.landmark[idx]
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        landmarks.extend([0.0] * len(upper_pose_indices) * 4)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        if len(hand_results.multi_hand_landmarks) == 1:
            landmarks.extend([0.0] * 21 * 3)
    else:
        landmarks.extend([0.0] * 21 * 3 * 2)
    arr = np.array(landmarks, dtype=np.float32)
    if pose_results.pose_landmarks:
        nose_x = arr[0]
        nose_y = arr[1]
        num_pose = len(upper_pose_indices)
        for i in range(num_pose):
            arr[i*4] = arr[i*4] - nose_x
            arr[i*4+1] = arr[i*4+1] - nose_y
        hand_offset = num_pose*4
        for h in range(2):
            for i in range(21):
                idx = hand_offset + h*21*3 + i*3
                arr[idx] = arr[idx] - nose_x
                arr[idx+1] = arr[idx+1] - nose_y
        xs = [arr[i*4] for i in range(num_pose)]
        ys = [arr[i*4+1] for i in range(num_pose)]
        for h in range(2):
            for i in range(21):
                idx = hand_offset + h*21*3 + i*3
                xs.append(arr[idx])
                ys.append(arr[idx+1])
        xs = np.array(xs)
        ys = np.array(ys)
        max_abs_x = np.max(np.abs(xs)) if np.max(np.abs(xs)) > 0 else 1.0
        max_abs_y = np.max(np.abs(ys)) if np.max(np.abs(ys)) > 0 else 1.0
        for i in range(num_pose):
            arr[i*4] = arr[i*4] / max_abs_x
            arr[i*4+1] = arr[i*4+1] / max_abs_y
        for h in range(2):
            for i in range(21):
                idx = hand_offset + h*21*3 + i*3
                arr[idx] = arr[idx] / max_abs_x
                arr[idx+1] = arr[idx+1] / max_abs_y
    return arr.tolist()

# Paths
ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), '../train/models/pose_classifier.pth')
input_size = 186
num_classes = 4

# Load model
model = PoseClassifier(input_size, num_classes)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

# List asset images
asset_files = [f for f in os.listdir(ASSETS_DIR) if f.lower().endswith('.jpg')]

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
window_name = 'Pose Game'
cv2.namedWindow(window_name)

while True:
    # Wait for 'f' key to start
    while True:
        img = np.zeros((480, 1280, 3), dtype=np.uint8)
        cv2.putText(img, 'Press F to start', (400, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 4)
        cv2.imshow(window_name, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'):
            break
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Randomly pick an image
    asset_file = random.choice(asset_files)
    asset_img = cv2.imread(os.path.join(ASSETS_DIR, asset_file))
    # Resize asset image with aspect ratio preserved and pad to 480x480
    def resize_and_pad(img, size=480):
        h, w = img.shape[:2]
        scale = size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (nw, nh))
        top = (size - nh) // 2
        bottom = size - nh - top
        left = (size - nw) // 2
        right = size - nw - left
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
        return img_padded
    asset_img = resize_and_pad(asset_img, 480)

    # Show asset and camera side by side
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5) as hands:
        countdown = 10
        start_time = None
        captured = False
        frame_landmarks = None
        result_time = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_flipped = cv2.flip(frame, 1)
            cam_img = resize_and_pad(frame_flipped, 480)
            image = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            pose_results = pose.process(image)
            hand_results = hands.process(image)
            image.flags.writeable = True
            cam_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Compose side by side
            combined = np.zeros((480, 960, 3), dtype=np.uint8)
            combined[:, :480] = asset_img
            combined[:, 480:] = cam_img

            if start_time is None:
                start_time = cv2.getTickCount()
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            remaining = int(countdown - elapsed)
            if remaining > 0 and not captured:
                cv2.putText(combined, f"Time: {remaining}s", (700, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)
            else:
                if not captured:
                    # Capture landmarks
                    frame_landmarks = extract_landmarks(pose_results, hand_results)
                    captured = True
                    result_time = cv2.getTickCount()
                    # Inference
                    if len(frame_landmarks) == input_size:
                        x = torch.tensor([frame_landmarks], dtype=torch.float32)
                        with torch.no_grad():
                            logits = model(x)
                            logits_np = logits.cpu().numpy().flatten()
                            pred_idx = int(np.argmax(logits_np))
                            class_map = ["biceps_flex", "cross_arms", "hands_up", "t_pose"]
                            shown_class = os.path.splitext(asset_file)[0]
                            if logits_np[pred_idx] > 0.8:
                                is_correct = (class_map[pred_idx] == shown_class)
                                acc_label = "Correct!" if is_correct else "Incorrect"
                                color = (0,255,0) if is_correct else (0,0,255)
                                probs = torch.softmax(torch.tensor(logits_np), dim=0).numpy()
                                percent = probs[pred_idx] * 100
                                cv2.putText(combined, f"Result: {acc_label} ({percent:.1f}%)", (600, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            else:
                                cv2.putText(combined, "None", (600, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    else:
                        cv2.putText(combined, "No/Partial Landmarks", (600, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                else:
                    # Show result for 5 seconds
                    show_result = True
                    if result_time is not None:
                        result_elapsed = (cv2.getTickCount() - result_time) / cv2.getTickFrequency()
                        if result_elapsed > 5:
                            show_result = False
                    if show_result:
                        if len(frame_landmarks) == input_size:
                            x = torch.tensor([frame_landmarks], dtype=torch.float32)
                            with torch.no_grad():
                                logits = model(x)
                                logits_np = logits.cpu().numpy().flatten()
                                pred_idx = int(np.argmax(logits_np))
                                class_map = ["biceps_flex", "cross_arms", "hands_up", "t_pose"]
                                shown_class = os.path.splitext(asset_file)[0]
                                if logits_np[pred_idx] > 0.8:
                                    is_correct = (class_map[pred_idx] == shown_class)
                                    acc_label = "Correct!" if is_correct else "Incorrect"
                                    color = (0,255,0) if is_correct else (0,0,255)
                                    probs = torch.softmax(torch.tensor(logits_np), dim=0).numpy()
                                    percent = probs[pred_idx] * 100
                                    cv2.putText(combined, f"Result: {acc_label} ({percent:.1f}%)", (600, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                                else:
                                    cv2.putText(combined, "None", (600, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                        else:
                            cv2.putText(combined, "No/Partial Landmarks", (600, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                    else:
                        cv2.putText(combined, "Press F to play again", (500, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow(window_name, combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('f') and captured and result_time is not None:
                result_elapsed = (cv2.getTickCount() - result_time) / cv2.getTickFrequency()
                if result_elapsed > 5:
                    break
            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                exit()
