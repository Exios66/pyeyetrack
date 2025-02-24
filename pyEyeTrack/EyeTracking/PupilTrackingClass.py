from .AbstractEyeTrackingClass import EyeTracking
import numpy as np
import pandas as pd
import cv2
import time
from pyEyeTrack.DataHandling import QueueHandling
from datetime import datetime
import os


class PupilTracking(EyeTracking):
    """
    A class for tracking pupils in video frames.
    """

    def __init__(self, source=0, session_id=None):
        """
        Initialize PupilTracking with video source and session ID
        Args:
            source: Camera index or video file path (default is 0 for primary webcam)
            session_id: Unique identifier for the tracking session (default is None)
        """
        super().__init__(source)
        self.session_id = session_id if session_id else datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data = {
            'Session_ID': [],
            'Time': [],
            'Left_Pupil_X': [],
            'Left_Pupil_Y': [],
            'Right_Pupil_X': [],
            'Right_Pupil_Y': []
        }
        self.eye_data_log = {"Timestamps": [], "Left_Eye_X": [], "Left_Eye_Y": []  
                             , "Right_Eye_X": [], "Right_Eye_Y": []}
        # dictionary to store the location of the pupil center and 
        # the corresponding timestamp
        self.queue_handler = QueueHandling()
        # intialized queue to do real-time data transfer

    def detect_eye(self, eye_points, facial_landmarks):
        """
        This function returns a numpy array of the x, y coordinates of the 
        landmarks that define the eye in the frame.

        Args:
            eye_points (list): the list of indicies of the facial landmarks 
            which represent an eye 
            facial_landmarks (dlib.full_object_detection): this object helps 
            get the location of the eye in the frame

        Returns:
            numpy array: the array of points that define the location of the 
            eye in the frame.
        """

        eye_landmarks_coordinates = np.array(
            [(facial_landmarks.part(eye_points[0]).x,
              facial_landmarks.part(eye_points[0]).y),
             (facial_landmarks.part(eye_points[1]).x,
              facial_landmarks.part(eye_points[1]).y),
             (facial_landmarks.part(eye_points[2]).x,
              facial_landmarks.part(eye_points[2]).y),
             (facial_landmarks.part(eye_points[3]).x,
              facial_landmarks.part(eye_points[3]).y),
             (facial_landmarks.part(eye_points[4]).x,
              facial_landmarks.part(eye_points[4]).y),
             (facial_landmarks.part(eye_points[5]).x,
              facial_landmarks.part(eye_points[5]).y)],
            np.int32)
        return eye_landmarks_coordinates

    def get_connected_components(self, thresholded_pupil_region):
        """
        This function returns the pupil ceter of the eye. 
        The input parameter is the thresholded pupil region.
        The pupil center is the centroid of the connected component
        with the largest area. Since we already have the approximate
        pupil area, we assume that the connected component with the 
        largest area to be the the pupil.

        Args:
            thresholded_pupil_region (numpy array): the approximate 
            pupil area after filtering and thresholding is applied.

        Returns:
            (float, float): a tuple with the x, y coordinate of the 
            pupil center.
        """

        _, _, stats, centroids = cv2.connectedComponentsWithStats(
            thresholded_pupil_region, 4)

        area = []
        index = 0
        for stat in stats:
            area.append((stat[4], index))
            index = index + 1

        maximum_area = max(area)
        index_of_maximum_area = maximum_area[1]

        pupil_center = centroids[index_of_maximum_area]

        return pupil_center

    def get_approximate_pupil_rectangle(
            self, eye_landmarks_coordinates, frame):
        """
        In this function we first find the minimum and maximum for 
        x coordinate of the location the eye and similarly for the y coordinate.
        Here we have altered the values such that after cropping the area would 
        give us only the region inside the eye. This is the approximately
        the region where the pupil lies.

        Args:
            eye_landmarks_coordinates (numpy array): array of the x,y 
            coordinates of the location the eye
            frame (numpy array): it is the frame in the video or captured 
            by the camera

        Returns:
            numpy array: the area of the eye cropped tightly
        """

        eye_landmark_min_x = np.min(eye_landmarks_coordinates[:, 0]) + 10
        eye_landmark_max_x = np.max(eye_landmarks_coordinates[:, 0]) - 10
        eye_landmark_min_y = np.min(eye_landmarks_coordinates[:, 1]) + 1
        eye_landmark_max_y = np.max(eye_landmarks_coordinates[:, 1]) - 1
        approximate_pupil_region = frame[eye_landmark_min_y: eye_landmark_max_y,
                                         eye_landmark_min_x: eye_landmark_max_x]

        return approximate_pupil_region

    def get_pupil_center_coordinates(
            self,
            eye_landmarks_coordinates,
            threshold,
            frame):
        """
        This function returns the pupil center for a single eye. First we acquire 
        the approximate region of the frame in which the pupil lies.
        Then we perform thresholding on this cropped part of the frame. 
        We then send this proceesed part to the get_connected_components function
        which returns the pupil center.

        Args:
            eye_landmarks_coordinates (numpy array): array of the x,y coordinates 
            of the location the eye
            threshold (int): the value that should be used for thresholding
            frame (numpy array): it is the frame in the video or captured by the camera

        Returns:
            (float, float): a tuple containing x and y coordinates of the pupil center.
        """

        approximate_pupil_region = self.get_approximate_pupil_rectangle(
            eye_landmarks_coordinates, frame)

        median_blur_filter = cv2.medianBlur(approximate_pupil_region, 5)
        _, thresholded_pupil_region = cv2.threshold(
            median_blur_filter, threshold, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return self.get_connected_components(thresholded_pupil_region)

    def detect_pupil(self, frame):
        """
        Detect pupils in the frame
        Args:
            frame: Input frame
        Returns:
            tuple: (processed frame, pupil coordinates)
        """
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return frame, None
            
        # Get the first face detected
        (x, y, w, h) = faces[0]
        face_gray = gray[y:y+h, x:x+w]
        
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Detect eyes in the face region
        eyes = self.eye_cascade.detectMultiScale(face_gray)
        
        if len(eyes) < 2:
            return frame, None
            
        # Sort eyes by x-coordinate to determine left and right
        eyes = sorted(eyes, key=lambda e: e[0])  # Sort by x coordinate
        left_eye, right_eye = eyes[:2]  # Get the leftmost and rightmost eyes
        
        # Adjust eye coordinates to frame coordinates
        left_eye = (left_eye[0] + x, left_eye[1] + y, left_eye[2], left_eye[3])
        right_eye = (right_eye[0] + x, right_eye[1] + y, right_eye[2], right_eye[3])
        
        # Draw eye regions
        cv2.rectangle(frame, (left_eye[0], left_eye[1]), 
                     (left_eye[0] + left_eye[2], left_eye[1] + left_eye[3]), (0, 255, 0), 2)
        cv2.rectangle(frame, (right_eye[0], right_eye[1]), 
                     (right_eye[0] + right_eye[2], right_eye[1] + right_eye[3]), (0, 255, 0), 2)
        
        # Detect pupils
        left_pupil = self._detect_pupil_in_region(gray, left_eye)
        right_pupil = self._detect_pupil_in_region(gray, right_eye)
        
        if left_pupil and right_pupil:
            # Draw pupil centers
            cv2.circle(frame, left_pupil, 2, (0, 0, 255), -1)
            cv2.circle(frame, right_pupil, 2, (0, 0, 255), -1)
            return frame, (left_pupil, right_pupil)
            
        return frame, None

    def _detect_pupil_in_region(self, frame, eye_region):
        """
        Detect pupil center in eye region
        Args:
            frame: Input frame
            eye_region: Tuple of (x, y, w, h) for eye region
        Returns:
            Tuple of (x, y) coordinates of pupil center in frame coordinates
        """
        (ex, ey, ew, eh) = eye_region
        eye = frame[ey:ey+eh, ex:ex+ew]
        
        if eye.size == 0:
            return None
            
        # Convert to grayscale if needed
        if len(eye.shape) == 3:
            eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)

        # Enhance contrast
        eye = cv2.equalizeHist(eye)
        
        # Threshold to isolate dark regions (pupil)
        _, thresh = cv2.threshold(eye, 45, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Get largest contour (likely the pupil)
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        
        if M["m00"] == 0:
            return None
            
        # Calculate pupil center in eye region coordinates
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Convert to frame coordinates
        frame_x = ex + cx
        frame_y = ey + cy
        
        return (frame_x, frame_y)

    def functionality(self, frame):
        """
        Process frame for pupil tracking
        Args:
            frame: Input frame
        """
        # This method is called by the parent class's start() method
        # The parent class handles face and eye detection
        # We just need to detect pupils and store the data
        processed_frame, pupil_coords = self.detect_pupil(frame)
        
        if pupil_coords:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            self.data['Session_ID'].append(self.session_id)
            self.data['Time'].append(timestamp)
            self.data['Left_Pupil_X'].append(pupil_coords[0][0])
            self.data['Left_Pupil_Y'].append(pupil_coords[0][1])
            self.data['Right_Pupil_X'].append(pupil_coords[1][0])
            self.data['Right_Pupil_Y'].append(pupil_coords[1][1])

    def csv_writer(self, data):
        """
        Write tracking data to a CSV file.
        Args:
            data: Dictionary containing tracking data
        """
        df = pd.DataFrame(data)
        
        # Create Output directory if it doesn't exist
        os.makedirs('Output', exist_ok=True)
        
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join('Output', f'eye_tracking_data_{timestamp}.csv')
        
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")

    def get_eye_coordinates(self, landmarks):
        """
        Extract eye coordinates from facial landmarks
        Args:
            landmarks: Facial landmarks from MediaPipe FaceMesh
        Returns:
            Dictionary containing coordinates for left and right eyes
        """
        # MediaPipe FaceMesh eye landmarks
        LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

        def get_eye_rect(eye_indices):
            points = np.array([(int(landmarks.landmark[idx].x * self.frame.shape[1]),
                              int(landmarks.landmark[idx].y * self.frame.shape[0]))
                             for idx in eye_indices])
            x, y = points[:, 0], points[:, 1]
            return {
                'left': min(x),
                'top': min(y),
                'right': max(x),
                'bottom': max(y)
            }

        left_eye = get_eye_rect(LEFT_EYE)
        right_eye = get_eye_rect(RIGHT_EYE)
        
        return {'left_eye': left_eye, 'right_eye': right_eye}
