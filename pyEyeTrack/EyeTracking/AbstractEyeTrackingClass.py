import cv2
import keyboard
from abc import ABC, abstractmethod
import numpy as np
import os

import sys
import os
import bz2
from functools import partial
from tqdm import tqdm


SHAPE_PREDICTOR_FNAME = 'shape_predictor_68_face_landmarks.dat'
SHAPE_PREDICTOR_BZ2_FNAME = SHAPE_PREDICTOR_FNAME + '.bz2'
SHAPE_PREDICTOR_URL = 'http://dlib.net/files/{}'.format(SHAPE_PREDICTOR_BZ2_FNAME)

def _download_file(url, out_path):
    try:
        from urllib import urlretrieve          # Python 2
    except ImportError:
        from urllib.request import urlretrieve  # Python 3

    def reporthook(t, b=1, bsize=1, tsize=None, last_b=[0]):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    
    with tqdm(unit='B', unit_scale=True, miniters=1, desc=out_path) as t:
        urlretrieve(url, filename=out_path, reporthook=partial(reporthook, t))

def _bz2_decompress_inplace(path, out_path):
    with open(path, 'rb') as source, open(out_path, 'wb') as dest:
        dest.write(bz2.decompress(source.read()))

def check():
    print("shape_predictor_68_face_landmarks.dat file is needed.")
    print("Press n -if you already have it and place it in the current folder")
    print("Press y -file will start downloading.")

    download_input = input()
    if download_input == 'y':
        script_path = os.path.dirname(os.path.abspath(__file__))

        _download_file(SHAPE_PREDICTOR_URL, SHAPE_PREDICTOR_BZ2_FNAME)
        _bz2_decompress_inplace(SHAPE_PREDICTOR_BZ2_FNAME,
                                    SHAPE_PREDICTOR_FNAME)


check()

class EyeTracking(ABC):
    """
    EyeTracking is an abstract class that is used to implement
    different types of eye-tracking events.
    In this library we have used this class to implement
    blink detection and pupil-tracking.

    Attributes:
        face_cascade: OpenCV face detector
        eye_cascade: OpenCV eye detector

    Methods:
        csv_writer(file_name)
            an abstract method that is to be used for
            .csv file generation.
        functionality(frame)
            an abstract method used to implement type of eye-tracking.
            e.g. blinking
        start()
            method to start eye-tracking
    """
    # Load OpenCV's pre-trained face and eye detection models
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def __init__(self, source):
        # acquire the webcam based on device id
        self.cap = cv2.VideoCapture(source)
        self.frame = None  # frame from the video or live-stream
        self.eyes = None  # variable to store eye regions
        self.face = None  # variable to store face region
        self.close_flag = False  # flag used to close the application

    @abstractmethod
    def csv_writer(self, file_name):
        """
        Implements writer to write the data dictionary to .csv file.

        Args:
            file_name (string): name of the .csv file to be generated.
        """
        pass

    @abstractmethod
    def functionality(self, frame):
        """
        Implement the eye-tracking functionality required.
        Args:
            frame (numpy array): it is the frame in the video or captured by
            the camera
        """
        pass

    def start(self):
        """
        This function reads the input from the video or the live-stream.
        It also processes the frame acquired and detects the face in the frame.
        Then all the facial landmarks are mapped to face detected in the frame.
        The frame and the facial landmarks are then used by the subclassed to
        implement blink detection or pupil tracking.
        The application terminates if the 'esc' key is pressed or if the
        close_flag is set to 'True'. If the face is not detected for 10 cycles
        of the loop the application will terminate.
        """
        face_not_detected = 0
        while True:
            if keyboard.is_pressed('esc') or self.close_flag or face_not_detected >= 10:
                break

            ret, self.frame = self.cap.read()

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Convert frame to grayscale
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                print('Face not detected. Find better lighting.')
                face_not_detected += 1
                continue

            face_not_detected = 0
            self.face = faces[0]  # Get the first face detected
            
            # Get the face region
            (x, y, w, h) = self.face
            face_gray = gray[y:y+h, x:x+w]
            
            # Detect eyes in the face region
            self.eyes = self.eye_cascade.detectMultiScale(face_gray)
            
            if len(self.eyes) >= 2:  # We need at least two eyes
                self.functionality(gray)
            else:
                print('Eyes not detected clearly. Please adjust lighting or position.')
                continue

            # Display the frame
            cv2.imshow('Eye Tracking', self.frame)
            if cv2.waitKey(1) == 27:  # Exit on ESC
                break

        self.cap.release()
        cv2.destroyAllWindows()
