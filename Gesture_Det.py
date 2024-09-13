import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
frame_height = 224
frame_width = 224


def preprocess_frame(frame):
    frame = cv2.resize(frame, (frame_width, frame_height))
    return frame


def extract_hand_landmarks(image, hand_landmarks):
    h, w, c = image.shape
    landmarks = []
    for hand_landmark in hand_landmarks:
        for lm in hand_landmark.landmark:
            landmarks.append((int(lm.x * w), int(lm.y * h), lm.z))  # Include z-coordinate
    return landmarks


def detect_index_finger_direction(landmarks):
    if len(landmarks) < 9:
        return "Unknown"

    wrist = landmarks[0]
    index_tip = landmarks[8]

    # Calculate the difference in positions
    dx = index_tip[0] - wrist[0]
    dy = index_tip[1] - wrist[1]
    dz = index_tip[2] - wrist[2]

    # Determine the direction based on the differences
    if abs(dx) > abs(dy) and abs(dx) > abs(dz):
        if dx > 0:
            return "Right"
        else:
            return "Left"
    elif abs(dy) > abs(dx) and abs(dy) > abs(dz):
        if dy > 0:
            return "Backward"
        else:
            return "Forward"
    #elif abs(dz) > max(abs(dx), abs(dy)):
        #if dz > 0:
         #   return "Forward"
        #else:
            #return "Backward"
    else:
        return "Unknown"


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        directions = []
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = extract_hand_landmarks(frame, [hand_landmarks])
            direction = detect_index_finger_direction(landmarks)
            directions.append(direction)

        for i, direction in enumerate(directions):
            cv2.putText(frame, f"Hand {i + 1} Direction: {direction}", (10, 30 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Index Finger Direction Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
