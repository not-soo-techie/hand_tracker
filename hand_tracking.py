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

prev_wrist_z = None
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

            # Compute wrist-relative coordinates
            wrist = landmarks[0]
            relative_landmarks = landmarks - wrist

            # 4️⃣ Print example relative value
            print("Index tip relative:", relative_landmarks[8])

            # Convert normalized coords to pixel coords
            pixel_landmarks = np.zeros((21, 2), dtype=int)

            for i in range(21):
                pixel_landmarks[i] = (
                    int(landmarks[i][0] * w),
                    int(landmarks[i][1] * h)
                )

            # Draw circles on each landmark
            for i, (x, y) in enumerate(pixel_landmarks):
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 255, 0), 1)

            print("Shape:", landmarks.shape)

            current_z = landmarks[0][2]

            z_buffer.append(current_z)

            # Compute pinch distance

            thumb_tip = relative_landmarks[4]
            index_tip = relative_landmarks[8]

            pinch_distance = np.linalg.norm(thumb_tip - index_tip)
            index_wrist_distance = np.linalg.norm(index_tip)

            if pinch_distance < 0.05 and index_wrist_distance > 0.2:
                cv2.putText(frame, "PINCH!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)



            # fist
                
            fist_count = 0

            for tip in [8, 12, 16, 20]:
                dist = np.linalg.norm(relative_landmarks[tip])
                if dist < 0.15:
                    fist_count += 1

            if fist_count == 4:
                cv2.putText(frame, "FIST!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2)

            # palm

            open_count = 0

            for tip in [8, 12, 16, 20]:
                dist = np.linalg.norm(relative_landmarks[tip])
                if dist > 0.25:
                    open_count += 1

            if open_count == 4:
                cv2.putText(frame, "OPEN PALM!", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

            # fixing punch/ z-index
            # current_z = landmarks[0][2]
            current_z = landmarks[8][2]

            


            z_buffer.append(current_z)

            if len(z_buffer) > BUFFER_SIZE:
                z_buffer.pop(0)


            if len(z_buffer) == BUFFER_SIZE:
                velocity = z_buffer[-2] - z_buffer[-1]
                print("velocity:", velocity)
                velocity = np.mean(np.diff(z_buffer))
                velocity = -velocity  # flip sign so forward = positive

                if velocity > 0.015:
                    print("Forward spike:", velocity)

                # if velocity > 0.02:   # tune this
                #     cv2.putText(frame, "FORWARD MOTION!", (50, 200),
                #                 cv2.FONT_HERSHEY_SIMPLEX,
                #                 1, (255, 255, 0), 2)


            # prev_wrist_z = current_wrist_z
            current_time = time.time()

            if fist_count == 4 and velocity > 0.02:
                if current_time - last_punch_time > PUNCH_COOLDOWN:
                    cv2.putText(frame, "PUNCH!", (50, 250),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 3)
                    last_punch_time = current_time

            # print("Wrist Z:", landmarks[0][2])
            # print("Index Z:", landmarks[8][2])  







    cv2.imshow("Gesture Arena - Modern API", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
