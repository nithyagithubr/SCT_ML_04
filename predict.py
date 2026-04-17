import cv2
import mediapipe as mp
import joblib

# Load model
model = joblib.load("gesture_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    gesture_text = "No Hand"

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            x_vals = []
            y_vals = []

            for lm in handLms.landmark:
                x_vals.append(lm.x)
                y_vals.append(lm.y)

            min_x = min(x_vals)
            min_y = min(y_vals)

            landmarks = []
            for lm in handLms.landmark:
                landmarks.append(lm.x - min_x)
                landmarks.append(lm.y - min_y)
               

            prediction = model.predict([landmarks])[0]
            print(prediction)

            gesture_text = f"Gesture: {prediction}"

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.putText(img, gesture_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()