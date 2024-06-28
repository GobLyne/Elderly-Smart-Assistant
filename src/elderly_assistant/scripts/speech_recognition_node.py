#!/usr/bin/env python

import rospy
import pygame
import speech_recognition as sr
from gtts import gTTS

# Initialize pygame mixer for sound
pygame.init()

mixerAlarm = pygame.mixer
mixerVoice = pygame.mixer

mixerAlarm.init()
mixerVoice.init()

def sound_alarm():
    mixerAlarm.music.load('/home/mustar/pro/catkin_ws/media/alarm_sound.mp3')
    mixerAlarm.music.play(loops=5)
    while mixerAlarm.music.get_busy():
        continue

def recognize_speech_and_detect_phrase():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    rospy.loginfo("Listening to you...")

    while not rospy.is_shutdown():
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            rospy.loginfo("Say something...")
            audio = recognizer.listen(source)

            try:
                rospy.loginfo("Recognizing audio...")
                recognized_audio = recognizer.recognize_google(audio)
                rospy.loginfo(f"Audio: {recognized_audio}")

                if "help" in recognized_audio.lower():
                    rospy.loginfo("Phrase detected! Sounding the alarm...")
                    sound_alarm()
                elif "where" in recognized_audio.lower():
                    words = recognized_audio.split()
                    user_input = words[-1]
                    rospy.loginfo(f"Phrase detected! Searching for the {user_input}...")

                    location = rospy.get_param(f'/object_locations/{user_input}', "not detected")
                    rospy.loginfo(f"The {user_input} is located at the {location}")
                    speak(f"The {user_input} is located at the {location}")

            except sr.UnknownValueError:
                rospy.loginfo("Could not understand the audio")
            except sr.RequestError as e:
                rospy.loginfo(f"Could not request results from the speech recognition service; {e}")

def speak(text):
    tts = gTTS(text=text, lang='en')
    filename = "/home/mustar/pro/catkin_ws/media/voice.mp3"

    tts.save(filename)
    mixerVoice.music.load(filename)
    mixerVoice.music.play()
    while mixerVoice.music.get_busy():
        continue
    mixerVoice.music.stop()
    mixerVoice.music.unload()

if __name__ == "__main__":
    rospy.init_node('speech_recognition_node', anonymous=True)
    recognize_speech_and_detect_phrase()

