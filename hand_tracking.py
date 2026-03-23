import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time

# Load model
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

z_buffer = []
BUFFER_SIZE = 10

last_punch_time = 0
PUNCH_COOLDOWN = 0.6

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    velocity = 0.0

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            h, w, _ = frame.shape

            landmarks = []
            for lm in hand_landmarks:
                landmarks.append([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks)

            # Wrist-relative coordinates
            wrist = landmarks[0]
            relative_landmarks = landmarks - wrist

            # Convert to pixel coords
            pixel_landmarks = np.zeros((21, 2), dtype=int)
            for i in range(21):
                pixel_landmarks[i] = (
                    int(landmarks[i][0] * w),
                    int(landmarks[i][1] * h)
                )

            # Draw landmarks
            for i, (x, y) in enumerate(pixel_landmarks):
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 255, 0), 1)

            # ---- PINCH ----
            thumb_tip = relative_landmarks[4]
            index_tip = relative_landmarks[8]

            pinch_distance = np.linalg.norm(thumb_tip - index_tip)
            index_wrist_distance = np.linalg.norm(index_tip)

            if pinch_distance < 0.05 and index_wrist_distance > 0.2:
                cv2.putText(frame, "PINCH!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)

            # ---- FIST ----
            fist_count = 0
            for tip in [8, 12, 16, 20]:
                dist = np.linalg.norm(relative_landmarks[tip])
                if dist < 0.18:  # slightly relaxed threshold
                    fist_count += 1

            # ---- OPEN PALM ----
            open_count = 0
            for tip in [8, 12, 16, 20]:
                dist = np.linalg.norm(relative_landmarks[tip])
                if dist > 0.25:
                    open_count += 1

            if open_count == 4:
                cv2.putText(frame, "OPEN PALM!", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

            # ---- Z VELOCITY (FIX: sirf ek baar append) ----
            current_z = landmarks[8][2]  # index finger tip ka z

            z_buffer.append(current_z)

            if len(z_buffer) > BUFFER_SIZE:
                z_buffer.pop(0)

            if len(z_buffer) == BUFFER_SIZE:
                diffs = np.diff(z_buffer)
                # Use abs max — works regardless of camera Z direction
                velocity = float(np.max(np.abs(diffs)))
                print("velocity:", round(velocity, 4))

            # ---- PUNCH DETECTION ----
            current_time = time.time()

            # Velocity is PRIMARY condition — fist is just a soft filter
            # If hand moves forward fast enough AND at least 2 fingers curled = PUNCH
            is_moving_forward = velocity > 0.012
            is_fist_like = fist_count >= 2  # relaxed — mid-punch fingers aren't fully curled

            if is_moving_forward and is_fist_like:
                if current_time - last_punch_time > PUNCH_COOLDOWN:
                    cv2.putText(frame, "PUNCH!", (50, 250),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (0, 0, 255), 3)
                    print(">>> PUNCH DETECTED! velocity:", round(velocity, 4))
                    last_punch_time = current_time

            # Show FIST separately only when NOT punching
            elif fist_count == 4 and not is_moving_forward:
                cv2.putText(frame, "FIST!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2)

            # Show velocity bar on screen (debug helper)
            bar_val = int(min(velocity * 3000, 200))
            cv2.rectangle(frame, (w - 30, h - 20), (w - 10, h - 20 - bar_val),
                          (0, 255, 255), -1)
            cv2.putText(frame, "vel", (w - 40, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    cv2.imshow("Gesture Arena", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()