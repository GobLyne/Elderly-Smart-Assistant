#!/usr/bin/env python

import rospy
import cv2
import cvzone
import math
import time
import mediapipe as mp
from ultralytics import YOLO

detected_objects = {}

def sound_alarm():
    mixerAlarm.music.load('/home/mustar/pro/catkin_ws/media/alarm_sound.mp3')
    mixerAlarm.music.play(loops=5)
    while mixerAlarm.music.get_busy():
        continue

def determine_location(detected_object, detected_objects):
    x1, y1 = detected_objects[detected_object][0]
    x2, y2 = detected_objects[detected_object][1]

    object_location = "center"

    for obj, bbox in detected_objects.items():
        if obj != detected_object:
            bx1, by1 = bbox[0]
            bx2, by2 = bbox[1]

            if y1 < by1 and y2 < by2:
                object_location = f"above {obj}"
                break
            elif y1 > by1 and y2 > by2:
                object_location = f"below {obj}"
                break
            elif x1 < bx1 and x2 < bx2:
                object_location = f"right of {obj}"
                break
            elif x1 > bx1 and x2 > bx2:
                object_location = f"left of {obj}"
                break
            else:
                dx = (x1 + x2) / 2 - (bx1 + bx2) / 2
                dy = (y1 + y2) / 2 - (by1 + by2) / 2
                distance = math.sqrt(dx ** 2 + dy ** 2)

                if distance < 100:
                    object_location = f"near {obj}"
                    break

    return object_location

def detect_fall():
    cap = cv2.VideoCapture(0)
    model = YOLO('yolov8n.pt')

    classnames = []
    with open('/home/mustar/pro/catkin_ws/media/classes.txt', 'r') as f:
        classnames = f.read().splitlines()

    fall_start_time = None
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    while not rospy.is_shutdown():
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

                cvzone.cornerRect(frame, [x1, y1, x2 - x1, y2 - y1], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_name} {conf}%', [x1 + 8, y1 - 12], thickness=2, scale=2)
                detected_objects[class_name] = [(x1, y1), (x2, y2)]

                if class_name.lower() == "person":
                    person_roi = frame[y1:y2, x1:x2]
                    person_roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                    result_pose = pose.process(person_roi_rgb)

                    if result_pose.pose_landmarks:
                        landmarks = result_pose.pose_landmarks.landmark
                        shoulder_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2
                        hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2

                        if shoulder_y > hip_y and conf > 80:
                            if fall_start_time is None:
                                fall_start_time = time.time()
                                rospy.loginfo("Fall start time recorded")
                            else:
                                fall_duration = time.time() - fall_start_time
                                rospy.loginfo(f"Fall duration: {fall_duration} seconds")
                                if fall_duration > 0:
                                    fall_detected = True
                                    cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 40], thickness=2, scale=2)
                                    rospy.loginfo("Fall detected! Sounding the alarm...")
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
    rospy.init_node('computer_vision_node', anonymous=True)
    detect_fall()

