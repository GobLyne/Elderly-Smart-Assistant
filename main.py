import cv2
import cvzone
import math
import threading
import pygame
import time
import speech_recognition as sr
from ultralytics import YOLO

# Initialize pygame mixer for sound
pygame.mixer.init()
pygame.mixer.music.load('media\\alarm sound.mp3')

# Function to play the alarm sound
def sound_alarm():
    pygame.mixer.music.play(loops=5)
    while pygame.mixer.music.get_busy():
        continue

def recognize_speech_and_detect_phrase():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("Listening for the phrase 'Please, help me!'...")

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

            except sr.UnknownValueError:
                print("Could not understand the audio")
            except sr.RequestError as e:
                print(f"Could not request results from the speech recognition service; {e}")

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (980, 740))

        results = model(frame)

        fall_detected = False

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

                # Perform fall detection only for persons (assuming class ID 0 is for persons)
                if class_name.lower() == "person":
                    height = y2 - y1
                    width = x2 - x1
                    threshold = height - width

                    if threshold < 0 and conf > 80:
                        if fall_start_time is None:
                            fall_start_time = time.time()
                            print("Fall start time recorded")
                        else:
                            fall_duration = time.time() - fall_start_time
                            print(f"Fall duration: {fall_duration} seconds")
                            if fall_duration > 3:
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