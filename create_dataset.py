import cv2
import mediapipe as mp
import os
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

# Your dataset path
dataset_path = r"C:\Users\DELL\SkillCraft Technology\hand_gesture_project\leapGestRecog"

output_file = "dataset.csv"

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)

    # Header
    header = []
    for i in range(21):
        header += [f'x{i}', f'y{i}']
    header.append("label")
    writer.writerow(header)

    # Loop through dataset
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)

        if not os.path.isdir(person_path):
            continue

        for gesture in os.listdir(person_path):

            print("Processing gesture:", gesture)  # ✅ debug

            gesture_path = os.path.join(person_path, gesture)

            if not os.path.isdir(gesture_path):
                continue

            for img_name in os.listdir(gesture_path):
                img_path = os.path.join(gesture_path, img_name)

                img = cv2.imread(img_path)
                if img is None:
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    for handLms in results.multi_hand_landmarks:

                        # Normalize landmarks
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

                        # Save to CSV
                        writer.writerow(landmarks + [gesture])

print("✅ Dataset created successfully!")