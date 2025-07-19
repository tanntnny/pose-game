import cv2
import time
import numpy as np
from utils.extract_landmark import draw_pose_and_hands
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

def extract_landmarks(pose_results, hand_results):
    """
    Extract pose and hand landmarks as a flat numpy array.
    Returns None if no pose or hand landmarks are detected.
    """
    landmarks = []

    # Only keep upper torso and arms (pose indices: 0, 11, 12, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
    # 0: nose, 11: left_shoulder, 12: right_shoulder, 23: left_hip, 24: right_hip,
    # 13: left_elbow, 14: right_elbow, 15: left_wrist, 16: right_wrist,
    # 17: left_pinky, 18: right_pinky, 19: left_index, 20: right_index, 21: left_thumb, 22: right_thumb
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

    return arr

def main():
    cap = cv2.VideoCapture(0)
    recorded_landmarks = []  # Accumulate all frames in a list

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import io

    prev_image = None
    prev_landmarks = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

        countdown = False
        countdown_start = 0

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
            image = draw_pose_and_hands(image, pose_results, hand_results)
            # Use flipped frame for preview
            preview_frame = frame_flipped


            key = cv2.waitKey(1) & 0xFF

            # Press 'f' to record, 'r' to remove last
            if not countdown and key == ord('f'):
                countdown = True
                countdown_start = time.time()

            if not countdown and key == ord('r'):
                if recorded_landmarks:
                    removed = recorded_landmarks.pop()
                    print(f"Removed last frame. Total frames: {len(recorded_landmarks)}")
                    if recorded_landmarks:
                        prev_landmarks = recorded_landmarks[-1].copy()
                        # Keep the previous image as is (optional: could store images in a list for full undo)
                    else:
                        prev_landmarks = None
                        prev_image = None

            if countdown:
                elapsed = time.time() - countdown_start
                remaining = int(3 - elapsed)
                if remaining > 0:
                    cv2.putText(image, f"Recording in {remaining}s", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                else:
                    # Record just one frame
                    landmarks = extract_landmarks(pose_results, hand_results)
                    recorded_landmarks.append(landmarks)
                    prev_image = preview_frame.copy()
                    prev_landmarks = landmarks.copy()
                    print(f"Frame recorded. Total frames: {len(recorded_landmarks)}")
                    countdown = False

            # Compose right panel if previous frame exists
            if prev_image is not None and prev_landmarks is not None:
                # Create matplotlib figure for landmarks
                fig, axs = plt.subplots(2, 1, figsize=(3, 6))
                axs[0].imshow(cv2.cvtColor(prev_image, cv2.COLOR_BGR2RGB))
                axs[0].set_title('Prev Frame')
                axs[0].axis('off')
                # Plot only x, y for upper torso/arms and hands (21+21) with different colors and legends
                pose_xs, pose_ys = [], []
                lh_xs, lh_ys = [], []
                rh_xs, rh_ys = [], []
                num_pose = 15  # upper_pose_indices length
                for i in range(num_pose):
                    pose_xs.append(prev_landmarks[i*4])
                    pose_ys.append(prev_landmarks[i*4+1])
                hand_offset = num_pose*4
                for i in range(21):
                    lh_xs.append(prev_landmarks[hand_offset + i*3])
                    lh_ys.append(prev_landmarks[hand_offset + i*3 + 1])
                for i in range(21):
                    rh_xs.append(prev_landmarks[hand_offset + 21*3 + i*3])
                    rh_ys.append(prev_landmarks[hand_offset + 21*3 + i*3 + 1])
                axs[1].scatter(pose_xs, pose_ys, c='b', label='Pose', s=20)
                axs[1].scatter(lh_xs, lh_ys, c='g', label='Left Hand', s=20)
                axs[1].scatter(rh_xs, rh_ys, c='r', label='Right Hand', s=20)
                axs[1].set_title('Landmarks (x, y)')
                axs[1].set_xlim(-1, 1)
                axs[1].set_ylim(1, -1)
                axs[1].set_aspect('equal')
                axs[1].legend(loc='upper right', fontsize='small')
                # Draw grid lines
                axs[1].set_xticks(np.linspace(-1, 1, 9))
                axs[1].set_yticks(np.linspace(-1, 1, 9))
                axs[1].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
                axs[1].spines['top'].set_visible(False)
                axs[1].spines['right'].set_visible(False)
                axs[1].spines['bottom'].set_visible(False)
                axs[1].spines['left'].set_visible(False)
                plt.tight_layout()
                canvas = FigureCanvas(fig)
                buf = io.BytesIO()
                canvas.print_rgba(buf)
                buf.seek(0)
                landmark_img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                landmark_img = landmark_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                landmark_img = cv2.cvtColor(landmark_img, cv2.COLOR_RGBA2BGR)
                plt.close(fig)
                # Combine current image and right panel
                h1, w1 = image.shape[:2]
                h2, w2 = landmark_img.shape[:2]
                panel = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
                panel[:h1, :w1] = image
                panel[:h2, w1:w1+w2] = landmark_img
                cv2.imshow('Recording Landmarks', panel)
            else:
                cv2.imshow('Recording Landmarks', image)

            if key == 27:  # ESC to exit
                break

    cap.release()
    cv2.destroyAllWindows()

    # Save all recorded frames as a single numpy array in a single run folder
    if recorded_landmarks:
        import os
        base_dir = 'train/saves'
        os.makedirs(base_dir, exist_ok=True)
        # Find next available run folder
        run_idx = 1
        while True:
            run_dir = os.path.join(base_dir, f'run{run_idx}')
            if not os.path.exists(run_dir):
                break
            run_idx += 1
        os.makedirs(run_dir, exist_ok=True)
        save_path = os.path.join(run_dir, 'recorded_landmarks.npy')
        landmarks_array = np.stack(recorded_landmarks)
        np.save(save_path, landmarks_array)
        print(f"Saved {len(recorded_landmarks)} frames to {save_path}")
        print(f"Recorded landmarks shape: {landmarks_array.shape}")
    else:
        print("No landmarks recorded.")

if __name__ == "__main__":
    main()