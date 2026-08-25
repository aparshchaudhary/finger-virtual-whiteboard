import cv2
import mediapipe as mp
import numpy as np
import time

# -------------------------
# MediaPipe Setup
# -------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -------------------------
# Camera
# -------------------------
cap = cv2.VideoCapture(0)

width = 1280
height = 720

cap.set(3, width)
cap.set(4, height)

canvas = np.zeros((height, width, 3), dtype=np.uint8)

prev_x = None
prev_y = None

clear_start = None

# -------------------------
# Finger Detection
# -------------------------
def finger_up(hand, tip, pip):
    return hand.landmark[tip].y < hand.landmark[pip].y

# -------------------------
# Main Loop
# -------------------------
while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    mode = "IDLE"

    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        # Finger States
        index_up = finger_up(hand, 8, 6)
        middle_up = finger_up(hand, 12, 10)
        ring_up = finger_up(hand, 16, 14)
        pinky_up = finger_up(hand, 20, 18)

        h, w, _ = frame.shape

        index_x = int(hand.landmark[8].x * w)
        index_y = int(hand.landmark[8].y * h)

        middle_x = int(hand.landmark[12].x * w)
        middle_y = int(hand.landmark[12].y * h)

        # ----------------------------------
        # DRAW MODE
        # Index only
        # ----------------------------------
        if index_up and not middle_up:

            mode = "DRAW"

            cv2.circle(
                frame,
                (index_x, index_y),
                10,
                (0, 255, 0),
                -1
            )

            if prev_x is None:
                prev_x = index_x
                prev_y = index_y

            cv2.line(
                canvas,
                (prev_x, prev_y),
                (index_x, index_y),
                (255, 255, 255),
                5
            )

            prev_x = index_x
            prev_y = index_y

            clear_start = None

        # ----------------------------------
        # ERASER MODE
        # Index + Middle
        # ----------------------------------
        elif index_up and middle_up:

            mode = "ERASE"

            cv2.circle(
                frame,
                (middle_x, middle_y),
                20,
                (0, 0, 255),
                -1
            )

            cv2.circle(
                canvas,
                (middle_x, middle_y),
                30,
                (0, 0, 0),
                -1
            )

            prev_x = None
            prev_y = None

            clear_start = None

        # ----------------------------------
        # CLEAR SCREEN
        # Open Palm 2 Seconds
        # ----------------------------------
        elif index_up and middle_up and ring_up and pinky_up:

            mode = "CLEAR HOLD"

            if clear_start is None:
                clear_start = time.time()

            elapsed = time.time() - clear_start

            cv2.putText(
                frame,
                f"Hold: {2-elapsed:.1f}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            if elapsed >= 2:
                canvas[:] = 0
                clear_start = None

            prev_x = None
            prev_y = None

        else:

            mode = "IDLE"

            prev_x = None
            prev_y = None

            clear_start = None

    else:

        prev_x = None
        prev_y = None

        clear_start = None

    # -------------------------
    # Merge Canvas
    # -------------------------
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(
        gray,
        20,
        255,
        cv2.THRESH_BINARY
    )

    mask_inv = cv2.bitwise_not(mask)

    frame_bg = cv2.bitwise_and(
        frame,
        frame,
        mask=mask_inv
    )

    canvas_fg = cv2.bitwise_and(
        canvas,
        canvas,
        mask=mask
    )

    final = cv2.add(frame_bg, canvas_fg)

    # -------------------------
    # UI
    # -------------------------
    cv2.putText(
        final,
        f"Mode: {mode}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.putText(
        final,
        "Draw=Index | Erase=Index+Middle | Palm 2s=Clear",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow("Finger Whiteboard", final)

    key = cv2.waitKey(1)

    if key == 27:
        break

    elif key == ord('c'):
        canvas[:] = 0

cap.release()
cv2.destroyAllWindows()