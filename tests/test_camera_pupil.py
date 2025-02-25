import unittest
import numpy as np
import cv2
import tempfile
import shutil
from unittest.mock import Mock, patch
from datetime import datetime
from pyEyeTrack.PyEyeTrackRunnerClass import PyEyeTrackRunner
from pyEyeTrack.EyeTracking.PupilTrackingClass import PupilTracking

class TestPyEyeTrackCameraBase(unittest.TestCase):
    """Base test class for camera-related tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests"""
        cls.test_dir = tempfile.mkdtemp()
        
        # Create test image
        cls.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(cls.test_image, (320, 240), 5, (255, 255, 255), -1)  # Add white dot
        
        # Mock camera
        cls.mock_cap = Mock()
        cls.mock_cap.read.return_value = (True, cls.test_image)
        cls.mock_cap.isOpened.return_value = True
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after all tests"""
        shutil.rmtree(cls.test_dir)

class TestPyEyeTrackCamera(TestPyEyeTrackCameraBase):
    """Test camera handling and initialization"""
    
    def setUp(self):
        """Set up each test"""
        self.tracker = PyEyeTrackRunner(
            participant_id="test_participant",
            session_id="test_session"
        )
    
    @patch('cv2.VideoCapture')
    def test_camera_initialization(self, mock_cv2_cap):
        """Test camera initialization"""
        mock_cv2_cap.return_value = self.mock_cap
        
        success = self.tracker.initialize_camera(0)
        self.assertTrue(success)
        self.assertEqual(self.tracker.current_camera, 0)
        
        # Test invalid camera
        mock_cv2_cap.return_value.isOpened.return_value = False
        success = self.tracker.initialize_camera(999)
        self.assertFalse(success)
    
    @patch('cv2.VideoCapture')
    def test_camera_detection(self, mock_cv2_cap):
        """Test camera detection"""
        mock_cv2_cap.return_value = self.mock_cap
        
        cameras = self.tracker.detect_cameras()
        self.assertIsInstance(cameras, list)
        self.assertTrue(len(cameras) >= 0)
    
    @patch('cv2.VideoCapture')
    def test_camera_cleanup(self, mock_cv2_cap):
        """Test camera cleanup"""
        mock_cv2_cap.return_value = self.mock_cap
        
        self.tracker.initialize_camera(0)
        self.tracker.cleanup()
        
        self.assertIsNone(self.tracker.cap)
        mock_cv2_cap.return_value.release.assert_called_once()

class TestPyEyeTrackPupilDetection(TestPyEyeTrackCameraBase):
    """Test pupil detection functionality"""
    
    def setUp(self):
        """Set up each test"""
        self.pupil_tracker = PupilTracking(source=0)
    
    @patch('cv2.VideoCapture')
    def test_pupil_detection(self, mock_cv2_cap):
        """Test pupil detection"""
        mock_cv2_cap.return_value = self.mock_cap
        
        # Test with valid frame
        frame, pupils = self.pupil_tracker.detect_pupil(self.test_image)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, self.test_image.shape)
        
        # Test with None frame
        frame, pupils = self.pupil_tracker.detect_pupil(None)
        self.assertIsNone(pupils)
        
        # Test with invalid frame
        invalid_frame = np.zeros((10, 10), dtype=np.uint8)  # Too small
        frame, pupils = self.pupil_tracker.detect_pupil(invalid_frame)
        self.assertIsNone(pupils)
    
    def test_pupil_validation(self):
        """Test pupil validation"""
        # Create test pupils
        valid_pupils = [(100, 100), (200, 200)]
        invalid_pupils = [(0, 0), (1000, 1000)]  # Out of normal range
        
        # Test valid pupils
        self.assertTrue(self.pupil_tracker.validate_pupils(valid_pupils))
        
        # Test invalid pupils
        self.assertFalse(self.pupil_tracker.validate_pupils(invalid_pupils))
        
        # Test None pupils
        self.assertFalse(self.pupil_tracker.validate_pupils(None))
    
    def test_frame_processing(self):
        """Test frame processing pipeline"""
        # Create test frame with face
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        face_roi = test_frame[140:340, 220:420]
        cv2.circle(face_roi, (100, 100), 100, (255, 255, 255), -1)  # Add face
        cv2.circle(face_roi, (80, 80), 5, (0, 0, 0), -1)  # Add left eye
        cv2.circle(face_roi, (120, 80), 5, (0, 0, 0), -1)  # Add right eye
        
        # Process frame
        processed_frame, pupils = self.pupil_tracker.detect_pupil(test_frame)
        
        self.assertIsNotNone(processed_frame)
        self.assertEqual(processed_frame.shape, test_frame.shape)

if __name__ == '__main__':
    unittest.main() 