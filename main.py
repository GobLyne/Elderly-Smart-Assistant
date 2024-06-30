import cv2
import cvzone
import math
import threading
import pygame
import time
# import os
import speech_recognition as sr
import mediapipe as mp
from ultralytics import YOLO
from gtts import gTTS

# Initialize pygame mixer for sound
pygame.init()

mixerAlarm = pygame.mixer
mixerVoice = pygame.mixer

mixerAlarm.init()
mixerVoice.init()

object_locations = {}
detected_objects = {}

# Function to play the alarm sound
def sound_alarm():
    mixerAlarm.music.load('media\\alarm sound.mp3')
    mixerAlarm.music.play(loops=5)
    while mixerAlarm.music.get_busy():
        continue

def recognize_speech_and_detect_phrase():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("Listening to you...")

    while True:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Say something...")
            audio = recognizer.listen(source)

            try:
                print("Recognizing audio...")
                recognized_audio = recognizer.recognize_google(audio)
                print(f"Audio: {recognized_audio}")

                if "help" in recognized_audio.lower():
                    print("Phrase detected! Sounding the alarm...")
                    sound_alarm()
                elif "where" in recognized_audio.lower():
                    words = recognized_audio.split()
                    user_input = words[-1]  # take the last word of the sentence
                    print(f"Phrase detected! Searching for the {user_input}...")

                    # Check if the item is detected
                    if user_input in detected_objects:
                        # Determine location of the detected object
                        object_locations[user_input] = determine_location(user_input, detected_objects)
                        print(f"The {user_input} is located at the {object_locations[user_input]}")

                        # Convert output to speech using text-to-speech
                        speak(f"The {user_input} is located at the {object_locations[user_input]}")
                    else:
                        print(f"The {user_input} is not detected")
                        speak(f"The {user_input} is not detected")

            except sr.UnknownValueError:
                print("Could not understand the audio")
            except sr.RequestError as e:
                print(f"Could not request results from the speech recognition service; {e}")

def speak(text):
    tts = gTTS(text=text, lang='en')
    filename = "media\\voice.mp3"

    tts.save(filename)
    mixerVoice.music.load(filename)

    mixerVoice.music.play()
    while mixerVoice.music.get_busy():
        continue

    mixerVoice.music.stop()  # Stop the music explicitly
    mixerVoice.music.unload()
    # print(mixerVoice.music.get_busy())
    # os.remove(filename)

def determine_location(detected_object, detected_objects):
    # Get the bounding box of the detected object
    x1, y1 = detected_objects[detected_object][0]
    x2, y2 = detected_objects[detected_object][1]

    # Initialize the location of the detected object
    object_location = "center"

    # Check if the detected object is near other objects
    for obj, bbox in detected_objects.items():
        if obj != detected_object:
            # Get the bounding box of the other object
            bx1, by1 = bbox[0]
            bx2, by2 = bbox[1]

            # Check if the detected object is above, below, left, or right of the other object
            if y1 < by1 and y2 < by2:  # above
                object_location = f"above {obj}"
                break
            elif y1 > by1 and y2 > by2:  # below
                object_location = f"below {obj}"
                break
            elif x1 < bx1 and x2 < bx2:  # left
                object_location = f"right of {obj}"
                break
            elif x1 > bx1 and x2 > bx2:  # right
                object_location = f"left of {obj}"
                break
            else:
                # Calculate the distance between the centers of the bounding boxes
                dx = (x1 + x2) / 2 - (bx1 + bx2) / 2
                dy = (y1 + y2) / 2 - (by1 + by2) / 2
                distance = math.sqrt(dx ** 2 + dy ** 2)

                if distance < 100:  # adjust this value based on your camera resolution and object sizes
                    object_location = f"near {obj}"
                    break

    return object_location

def detect_fall():
    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Load YOLO model
    model = YOLO('yolov8s.pt')

    # Load class names
    classnames = []
    with open('classes.txt', 'r') as f:
        classnames = f.read().splitlines()

    fall_start_time = None

    # Initialize MediaPipe pose detection
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (980, 740))

        results = model(frame)

        fall_detected = False
        detected_objects.clear()

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_detect = int(box.cls[0])
                class_name = classnames[class_detect]
                conf = math.ceil(confidence * 100)

                # Draw bounding box and label for all detected objects
                cvzone.cornerRect(frame, [x1, y1, x2 - x1, y2 - y1], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_name} {conf}%', [x1 + 8, y1 - 12], thickness=2, scale=2)

                # Store detected object and its bounding box
                detected_objects[class_name] = [(x1, y1), (x2, y2)]

                # Perform fall detection only for persons (assuming class ID 0 is for persons)
                if class_name.lower() == "person":

                    # Extract ROI for pose estimation
                    person_roi = frame[y1:y2, x1:x2]
                    person_roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                    result_pose = pose.process(person_roi_rgb)

                    if result_pose.pose_landmarks:
                        # Extract key points
                        landmarks = result_pose.pose_landmarks.landmark
                        shoulder_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2
                        hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2

                        # Simple heuristic for fall detection: If the shoulders are significantly lower than the hips
                        if shoulder_y > hip_y and conf > 80:
                            if fall_start_time is None:
                                fall_start_time = time.time()
                                print("Fall start time recorded")
                            else:
                                fall_duration = time.time() - fall_start_time
                                print(f"Fall duration: {fall_duration} seconds")
                                if fall_duration > 0:
                                    fall_detected = True
                                    cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 40], thickness=2, scale=2)
                                    print("Fall detected! Sounding the alarm...")
                                    sound_alarm()
                        else:
                            fall_start_time = None

        if fall_detected:
            fall_start_time = None

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('t'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create threads for speech recognition and fall detection
    speech_thread = threading.Thread(target=recognize_speech_and_detect_phrase)
    fall_thread = threading.Thread(target=detect_fall)

    # Start the threads
    speech_thread.start()
    fall_thread.start()

    # Wait for both threads to complete
    speech_thread.join()
    fall_thread.join()