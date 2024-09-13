# Gesture-Driven-Car
### Gesture-Driven Car Control Using Hand Gesture Recognition

This project utilizes hand gesture recognition to control a car using gestures detected via a webcam. The implementation leverages the MediaPipe library to track hand landmarks and determine the direction of the index finger to interpret various gestures. Below is a detailed overview of the project, including setup instructions and explanations of the core functionality.

---

## Project Overview

The system processes real-time video input from a webcam to recognize hand gestures and determine the direction of the index finger. This information can be used to control various aspects of a car or other applications requiring gesture-based input.

### Key Features

- **Real-Time Hand Detection**: Utilizes MediaPipe's hand tracking to detect up to two hands and their landmarks in real-time.
- **Index Finger Direction Detection**: Determines the direction of the index finger relative to the wrist to interpret gestures.
- **Visualization**: Draws hand landmarks and gesture directions on the video feed for user feedback.

### Requirements

- Python 3.x
- OpenCV
- MediaPipe

You can install the necessary Python packages using pip:

```bash
pip install opencv-python mediapipe
```

### Usage

1. **Run the Script**:
   To start the gesture recognition, run the following command in your terminal:

   ```bash
   python gesture_control.py
   ```

2. **Interaction**:
   - **Press 'q'**: Exit the application.

### Code Explanation

#### Imports and Initialization

```python
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
```

- **`cv2`**: OpenCV for handling video capture and image processing.
- **`mediapipe`**: Library for hand landmark detection.
- **`mp_hands`**: MediaPipe hands solution.
- **`hands`**: Hand tracking model initialization.
- **`mp_drawing`**: Utilities for drawing hand landmarks.

#### Frame Preprocessing

```python
def preprocess_frame(frame):
    frame = cv2.resize(frame, (frame_width, frame_height))
    return frame
```

- Resizes frames to a fixed size to standardize input for processing.

#### Hand Landmark Extraction

```python
def extract_hand_landmarks(image, hand_landmarks):
    h, w, c = image.shape
    landmarks = []
    for hand_landmark in hand_landmarks:
        for lm in hand_landmark.landmark:
            landmarks.append((int(lm.x * w), int(lm.y * h), lm.z))  # Include z-coordinate
    return landmarks
```

- Extracts and normalizes hand landmarks (including the z-coordinate) from the image.

#### Index Finger Direction Detection

```python
def detect_index_finger_direction(landmarks):
    if len(landmarks) < 9:
        return "Unknown"

    wrist = landmarks[0]
    index_tip = landmarks[8]

    dx = index_tip[0] - wrist[0]
    dy = index_tip[1] - wrist[1]
    dz = index_tip[2] - wrist[2]

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
    else:
        return "Unknown"
```

- Determines the direction of the index finger by calculating the differences between the wrist and the index tip coordinates.

#### Main Loop

```python
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
```

- **Video Capture**: Captures video from the webcam and processes each frame.
- **Hand Tracking**: Processes the frame to detect hand landmarks.
- **Direction Detection**: Determines the direction of the index finger.
- **Visualization**: Draws landmarks and directions on the frame.
- **Exit Condition**: Exits the loop and closes the window when 'q' is pressed.

### Future Enhancements

- **Gesture Mapping**: Map detected directions to specific car controls.
- **Extended Gesture Set**: Include additional gestures for more control options.
- **Performance Optimization**: Enhance the performance for faster and more accurate gesture recognition.

Feel free to contribute to the project or provide feedback via issues or pull requests on the GitHub repository.

---

This README provides a comprehensive overview of the gesture-driven car control project, including setup instructions, code functionality, and potential future enhancements.
