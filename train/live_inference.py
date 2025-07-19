import os
import cv2
import mediapipe as mp
import torch
import numpy as np
from utils.classifier import PoseClassifier

# Paths
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'models/pose_classifier.pth')

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Load model
# You must set input_size and num_classes to match your training
input_size = 186  # 15 pose points * 4 + 21*3*2 hand points
num_classes = 4  # Update if you have a different number of classes
model = PoseClassifier(input_size, num_classes)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

# Helper to extract pose+hand landmarks as a flat vector
def extract_landmarks(pose_results, hand_results):
    import numpy as np
    landmarks = []
    upper_pose_indices = [0, 11, 12, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    if pose_results.pose_landmarks:
        for idx in upper_pose_indices:
            lm = pose_results.pose_landmarks.landmark[idx]
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        landmarks.extend([0.0] * len(upper_pose_indices) * 4)

    # Hand landmarks (21 points per hand, each with x, y, z)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        # If only one hand detected, pad the other
        if len(hand_results.multi_hand_landmarks) == 1:
            landmarks.extend([0.0] * 21 * 3)
    else:
        landmarks.extend([0.0] * 21 * 3 * 2)

    arr = np.array(landmarks, dtype=np.float32)

    # Shift all x, y so that nose is at (0,0), then normalize to [-1, 1]
    if pose_results.pose_landmarks:
        nose_x = arr[0]
        nose_y = arr[1]
        num_pose = len(upper_pose_indices)
        # Shift pose
        for i in range(num_pose):
            arr[i*4] = arr[i*4] - nose_x
            arr[i*4+1] = arr[i*4+1] - nose_y
        # Shift hands
        hand_offset = num_pose*4
        for h in range(2):
            for i in range(21):
                idx = hand_offset + h*21*3 + i*3
                arr[idx] = arr[idx] - nose_x
                arr[idx+1] = arr[idx+1] - nose_y

        # Gather all x and y after shifting
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
        # Normalize to [-1, 1]
        for i in range(num_pose):
            arr[i*4] = arr[i*4] / max_abs_x
            arr[i*4+1] = arr[i*4+1] / max_abs_y
        for h in range(2):
            for i in range(21):
                idx = hand_offset + h*21*3 + i*3
                arr[idx] = arr[idx] / max_abs_x
                arr[idx+1] = arr[idx+1] / max_abs_y

    return arr.tolist()

# Inference loop
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5) as hands:
    buffer = []
    THRESHOLD = 0.8
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_flipped = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        pose_results = pose.process(image)
        hand_results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        landmarks = extract_landmarks(pose_results, hand_results)
        if len(landmarks) == input_size:
            x = torch.tensor([landmarks], dtype=torch.float32)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, 1)
                if conf.item() > THRESHOLD:
                    buffer.append(pred.item())
                    label = f"Class: {pred.item()} ({conf.item():.2f})"
                    color = (0,255,0)
                else:
                    buffer.append(-1)
                    label = "None"
                    color = (0,0,255)
                cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            buffer.append(-1)
            cv2.putText(image, "No/Partial Landmarks", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow('Pose Inference', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

# Post-process buffer if needed (e.g., majority vote, smoothing)
