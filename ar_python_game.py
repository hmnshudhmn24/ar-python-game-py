import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Video Capture
cap = cv2.VideoCapture(0)

# Game Variables
target_pos = np.array([300, 200])  # Initial target position
score = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)  # Mirror the image
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get index finger tip position
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            
            # Draw finger position
            cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)
            
            # Check collision with target
            if np.linalg.norm(target_pos - np.array([x, y])) < 30:
                score += 1
                target_pos = np.random.randint(100, 500, size=2)  # New target position

    # Draw target
    cv2.circle(frame, tuple(target_pos), 20, (0, 255, 0), -1)
    
    # Display score
    cv2.putText(frame, f"Score: {score}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow("AR Game", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
