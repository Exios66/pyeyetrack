from .AbstractEyeTrackingClass import EyeTracking
import numpy as np
import pandas as pd
import cv2
import time
from pyEyeTrack.DataHandling import QueueHandling
from datetime import datetime
import os
import logging
from typing import Optional, Tuple, Dict, Any
import json

logger = logging.getLogger('PyEyeTrack.PupilTracking')

class PupilTracking(EyeTracking):
    """
    A class for tracking pupils in video frames with enhanced detection and validation.
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
        
        # Enhanced data storage
        self.data = {
            'Session_ID': [],
            'Time': [],
            'Left_Pupil_X': [],
            'Left_Pupil_Y': [],
            'Right_Pupil_X': [],
            'Right_Pupil_Y': [],
            'Left_Pupil_Size': [],
            'Right_Pupil_Size': [],
            'Left_Blink': [],
            'Right_Blink': [],
            'Confidence': []
        }
        
        # Tracking parameters
        self.min_pupil_size = 3  # minimum pupil size in pixels
        self.max_pupil_size = 50  # maximum pupil size in pixels
        self.min_confidence = 0.6  # minimum confidence threshold
        self.blink_threshold = 0.3  # threshold for blink detection
        
        # Kalman filter for smoothing
        self.kalman_left = cv2.KalmanFilter(4, 2)
        self.kalman_right = cv2.KalmanFilter(4, 2)
        self.setup_kalman_filters()
        
        logger.info(f"PupilTracking initialized for session {self.session_id}")

    def setup_kalman_filters(self):
        """Initialize Kalman filters for pupil tracking"""
        for kalman in [self.kalman_left, self.kalman_right]:
            kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
            kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                              [0, 1, 0, 1],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32)
            kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32) * 0.03

    def detect_blink(self, eye_region: np.ndarray) -> bool:
        """
        Detect if eye is blinking using intensity analysis
        Args:
            eye_region: Grayscale image of eye region
        Returns:
            bool: True if eye is blinking
        """
        if eye_region is None or eye_region.size == 0:
            return True
            
        # Calculate average intensity
        avg_intensity = np.mean(eye_region)
        
        # If intensity is high (bright), likely a blink
        return avg_intensity > (255 * self.blink_threshold)

    def calculate_pupil_size(self, thresh_eye: np.ndarray) -> float:
        """
        Calculate pupil size from thresholded eye image
        Args:
            thresh_eye: Binary image of eye region
        Returns:
            float: Pupil size in pixels
        """
        if thresh_eye is None or thresh_eye.size == 0:
            return 0.0
            
        # Find contours
        contours, _ = cv2.findContours(thresh_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
            
        # Get largest contour (pupil)
        pupil_contour = max(contours, key=cv2.contourArea)
        return cv2.contourArea(pupil_contour)

    def _detect_pupil_in_region(self, frame: np.ndarray, eye_region: Tuple[int, int, int, int]) -> Optional[Tuple[int, int]]:
        """
        Enhanced pupil detection in eye region with validation
        Args:
            frame: Input frame
            eye_region: Tuple of (x, y, w, h) for eye region
        Returns:
            Optional[Tuple[int, int]]: Pupil center coordinates or None if detection fails
        """
        try:
            (ex, ey, ew, eh) = eye_region
            eye = frame[ey:ey+eh, ex:ex+ew]
            
            if eye.size == 0:
                return None
                
            # Convert to grayscale if needed
            if len(eye.shape) == 3:
                eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)

            # Check for blink
            if self.detect_blink(eye):
                return None

            # Enhance contrast
            eye = cv2.equalizeHist(eye)
            
            # Apply bilateral filter to reduce noise while preserving edges
            eye = cv2.bilateralFilter(eye, 5, 75, 75)
            
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(eye, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
            
            # Morphological operations to clean up noise
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
                
            # Get largest contour (likely the pupil)
            pupil_contour = max(contours, key=cv2.contourArea)
            
            # Validate pupil size
            pupil_size = cv2.contourArea(pupil_contour)
            if not (self.min_pupil_size <= pupil_size <= self.max_pupil_size):
                return None
            
            # Calculate pupil center using moments
            M = cv2.moments(pupil_contour)
            if M["m00"] == 0:
                return None
                
            # Calculate pupil center in eye region coordinates
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Convert to frame coordinates
            frame_x = ex + cx
            frame_y = ey + cy
            
            return (frame_x, frame_y)
            
        except Exception as e:
            logger.error(f"Error in pupil detection: {str(e)}")
            return None

    def detect_pupil(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """
        Enhanced pupil detection with validation and tracking
        Args:
            frame: Input frame
        Returns:
            Tuple[np.ndarray, Optional[Tuple[Tuple[int, int], Tuple[int, int]]]]: 
            Processed frame and pupil coordinates
        """
        try:
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                logger.debug("No face detected")
                return frame, None
                
            # Get the first face detected
            (x, y, w, h) = faces[0]
            face_gray = gray[y:y+h, x:x+w]
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Detect eyes in the face region
            eyes = self.eye_cascade.detectMultiScale(face_gray)
            
            if len(eyes) < 2:
                logger.debug("Less than 2 eyes detected")
                return frame, None
                
            # Sort eyes by x-coordinate to determine left and right
            eyes = sorted(eyes, key=lambda e: e[0])
            left_eye, right_eye = eyes[:2]
            
            # Adjust eye coordinates to frame coordinates
            left_eye = (left_eye[0] + x, left_eye[1] + y, left_eye[2], left_eye[3])
            right_eye = (right_eye[0] + x, right_eye[1] + y, right_eye[2], right_eye[3])
            
            # Draw eye regions
            cv2.rectangle(frame, (left_eye[0], left_eye[1]), 
                         (left_eye[0] + left_eye[2], left_eye[1] + left_eye[3]), (0, 255, 0), 2)
            cv2.rectangle(frame, (right_eye[0], right_eye[1]), 
                         (right_eye[0] + right_eye[2], right_eye[1] + right_eye[3]), (0, 255, 0), 2)
            
            # Detect pupils with Kalman filtering
            left_pupil = self._detect_pupil_in_region(gray, left_eye)
            right_pupil = self._detect_pupil_in_region(gray, right_eye)
            
            if left_pupil and right_pupil:
                # Update Kalman filters
                left_measured = np.array([[np.float32(left_pupil[0])], [np.float32(left_pupil[1])]])
                right_measured = np.array([[np.float32(right_pupil[0])], [np.float32(right_pupil[1])]])
                
                self.kalman_left.correct(left_measured)
                self.kalman_right.correct(right_measured)
                
                left_predicted = self.kalman_left.predict()
                right_predicted = self.kalman_right.predict()
                
                # Use predicted values for smooth tracking
                left_pupil = (int(left_predicted[0]), int(left_predicted[1]))
                right_pupil = (int(right_predicted[0]), int(right_predicted[1]))
                
                # Draw pupil centers
                cv2.circle(frame, left_pupil, 2, (0, 0, 255), -1)
                cv2.circle(frame, right_pupil, 2, (0, 0, 255), -1)
                
                return frame, (left_pupil, right_pupil)
            
            return frame, None
            
        except Exception as e:
            logger.error(f"Error in pupil detection: {str(e)}")
            return frame, None

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

    def csv_writer(self, data: Dict[str, Any]) -> None:
        """
        Write tracking data to a CSV file with enhanced organization
        Args:
            data: Dictionary containing tracking data
        """
        try:
            df = pd.DataFrame(data)
            
            # Create Output directory if it doesn't exist
            os.makedirs('Output', exist_ok=True)
            
            # Generate timestamp for unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = os.path.join('Output', f'eye_tracking_data_{timestamp}.csv')
            
            # Calculate and add statistics
            stats = {
                'mean_confidence': np.mean(data['Confidence']),
                'blink_rate': sum(data['Left_Blink']) / len(data['Left_Blink']),
                'tracking_duration': (pd.to_datetime(data['Time'][-1]) - pd.to_datetime(data['Time'][0])).total_seconds()
            }
            
            # Save data and stats
            df.to_csv(file_path, index=False)
            
            # Save statistics separately
            stats_file = os.path.join('Output', f'tracking_stats_{timestamp}.json')
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=4)
            
            logger.info(f"Data saved to {file_path}")
            logger.info(f"Statistics saved to {stats_file}")
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")

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
