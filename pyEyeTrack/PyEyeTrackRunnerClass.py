import cv2
import numpy as np
import os
import threading
from datetime import datetime
import sys
import pandas as pd
import json
import time
import logging
from typing import Optional, Dict, Any, Tuple
from .AudioVideoRecording.VideoRecordingClass import VideoRecorder
try:
    from .AudioVideoRecording.AudioRecordingClass import AudioRecorder
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: PyAudio not available. Audio recording will be disabled.")

from .EyeTracking.PupilTrackingClass import PupilTracking

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PyEyeTrack')

class PyEyeTrackRunner:
    def __init__(self, participant_id=None, session_id=None, metadata=None):
        """
        Initialize the PyEyeTrackRunner class.
        Args:
            participant_id (str): Unique identifier for the participant
            session_id (str): Unique identifier for the session
            metadata (dict): Additional session metadata
        """
        self.running = True
        self.recording = False
        self.paused = False
        self.window_name = "PyEyeTrack"
        self.participant_id = participant_id or "unknown"
        self.session_id = session_id or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.metadata = metadata or {}
        self.session_dir = None
        self.data_dir = None
        self.markers = []
        self.notes = []
        self.current_camera = None
        self.cap = None
        
        # Performance monitoring
        self.frame_times = []
        self.processing_times = []
        self.last_frame_time = None
        self.fps_update_interval = 1.0  # Update FPS every second
        self.last_fps_update = time.time()
        self.current_fps = 0.0
        
        # Initialize data storage with validation ranges
        self.eye_data = {
            'timestamp': [],
            'frame_number': [],
            'left_eye_x': [],
            'left_eye_y': [],
            'right_eye_x': [],
            'right_eye_y': [],
            'left_pupil_size': [],
            'right_pupil_size': [],
            'left_blink': [],
            'right_blink': [],
            'gaze_x': [],
            'gaze_y': [],
            'head_pose_x': [],
            'head_pose_y': [],
            'head_pose_z': [],
            'marker': [],
            'data_quality': []  # New field for data quality assessment
        }
        
        # Data validation ranges
        self.validation_ranges = {
            'pupil_size': (2.0, 8.0),  # mm
            'gaze_x': (0, 1920),  # pixels
            'gaze_y': (0, 1080),  # pixels
            'head_pose': (-90, 90)  # degrees
        }
        
        self.frame_count = 0
        self.total_frames = 0
        self.dropped_frames = 0
        self.start_time = None
        
        # Initialize calibration state
        self.is_calibrated = False
        self.calibration_points = []
        
        logger.info(f"PyEyeTrack initialized for participant {self.participant_id}, session {self.session_id}")

    def update_performance_metrics(self) -> None:
        """Update performance metrics including FPS calculation"""
        current_time = time.time()
        
        if self.last_frame_time is not None:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
            
            # Keep only last 100 frame times for memory efficiency
            if len(self.frame_times) > 100:
                self.frame_times.pop(0)
            
            # Update FPS every second
            if current_time - self.last_fps_update >= self.fps_update_interval:
                self.current_fps = len(self.frame_times) / sum(self.frame_times)
                self.last_fps_update = current_time
                logger.debug(f"Current FPS: {self.current_fps:.2f}")
        
        self.last_frame_time = current_time

    def validate_eye_data(self, eye_data: Dict[str, Any]) -> float:
        """
        Validate eye tracking data quality
        Returns quality score between 0 and 1
        """
        quality_score = 1.0
        deductions = []
        
        # Check pupil size
        if eye_data['left_pupil_size'] > 0:  # Only check if data exists
            if not (self.validation_ranges['pupil_size'][0] <= eye_data['left_pupil_size'] <= self.validation_ranges['pupil_size'][1]):
                quality_score -= 0.2
                deductions.append('pupil_size_out_of_range')
        
        # Check gaze coordinates
        if eye_data['gaze_x'] > 0 and eye_data['gaze_y'] > 0:
            if not (0 <= eye_data['gaze_x'] <= self.validation_ranges['gaze_x'][1]):
                quality_score -= 0.2
                deductions.append('gaze_x_out_of_range')
            if not (0 <= eye_data['gaze_y'] <= self.validation_ranges['gaze_y'][1]):
                quality_score -= 0.2
                deductions.append('gaze_y_out_of_range')
        
        # Check head pose
        for pose in ['head_pose_x', 'head_pose_y', 'head_pose_z']:
            if eye_data[pose] != 0:  # Only check if data exists
                if not (self.validation_ranges['head_pose'][0] <= eye_data[pose] <= self.validation_ranges['head_pose'][1]):
                    quality_score -= 0.1
                    deductions.append(f'{pose}_out_of_range')
        
        # Log quality issues
        if deductions:
            logger.warning(f"Data quality issues detected: {', '.join(deductions)}")
        
        return max(0.0, quality_score)

    def create_session_directory(self, base_dir):
        """Create a new session directory with timestamp."""
        self.session_dir = os.path.join(base_dir, f"data")
        self.data_dir = os.path.join(self.session_dir, "raw_data")
        os.makedirs(self.data_dir, exist_ok=True)
        return self.session_dir

    def add_marker(self, marker_text=None):
        """Add a marker/note to the current recording."""
        if not marker_text:
            marker_text = input("\nEnter marker note: ").strip()
        
        timestamp = datetime.now()
        marker = {
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'),
            'frame_number': self.frame_count,
            'note': marker_text
        }
        self.markers.append(marker)
        print(f"Marker added at frame {self.frame_count}: {marker_text}")

    def add_note(self, note_text):
        """Add a note without marking a specific frame"""
        timestamp = datetime.now()
        note = {
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'),
            'note': note_text
        }
        self.notes.append(note)
        print(f"Note added: {note_text}")

    def save_eye_data(self):
        """Save eye tracking data to CSV file with enhanced organization."""
        if len(self.eye_data['timestamp']) > 0:
            # Create DataFrame with all collected metrics
            df = pd.DataFrame(self.eye_data)
            
            # Add recording statistics
            stats = {
                'total_frames': self.total_frames,
                'dropped_frames': self.dropped_frames,
                'frame_rate': self.total_frames / (time.time() - self.start_time) if self.start_time else 0,
                'recording_duration': time.time() - self.start_time if self.start_time else 0,
                'camera_id': self.current_camera
            }
            
            # Save raw data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_filename = f"eye_tracking_data_{timestamp}.csv"
            csv_path = os.path.join(self.data_dir, csv_filename)
            df.to_csv(csv_path, index=False)
            
            # Save markers
            if self.markers:
                markers_file = os.path.join(self.data_dir, f"markers_{timestamp}.json")
                with open(markers_file, 'w') as f:
                    json.dump(self.markers, f, indent=4)
            
            # Save notes
            if self.notes:
                notes_file = os.path.join(self.data_dir, f"notes_{timestamp}.json")
                with open(notes_file, 'w') as f:
                    json.dump(self.notes, f, indent=4)
            
            # Save statistics
            stats_file = os.path.join(self.data_dir, f"recording_stats_{timestamp}.json")
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=4)
            
            print(f"\nRecording Summary:")
            print(f"Total Frames: {stats['total_frames']}")
            print(f"Dropped Frames: {stats['dropped_frames']} ({(stats['dropped_frames']/stats['total_frames']*100):.2f}%)")
            print(f"Average Frame Rate: {stats['frame_rate']:.2f} fps")
            print(f"Recording Duration: {stats['recording_duration']:.2f} seconds")
            print(f"Data saved to {csv_path}")
            
            # Clear the data for next recording
            for key in self.eye_data:
                self.eye_data[key] = []
            self.frame_count = 0
            self.total_frames = 0
            self.dropped_frames = 0
            self.start_time = None
            self.markers = []
            self.notes = []

    def process_key(self, key):
        """Process a key press event."""
        if key == ord('q'):
            self.running = False
            return True
        elif key == ord('r'):
            if self.recording:
                self.save_eye_data()
            else:
                self.start_time = time.time()
            self.recording = not self.recording
            status = "Started" if self.recording else "Stopped"
            print(f"\nRecording {status}")
            return True
        elif key == ord('s'):
            self.add_marker()
            return True
        elif key == ord('p'):
            self.paused = not self.paused
            status = "Paused" if self.paused else "Resumed"
            print(f"\nTracking {status}")
            return True
        return False

    def cleanup(self) -> None:
        """Clean up resources and save any remaining data"""
        try:
            if self.recording:
                self.save_eye_data()
            
            if self.cap is not None:
                self.cap.release()
            
            cv2.destroyAllWindows()
            
            # Save performance metrics
            if self.frame_times:
                avg_fps = len(self.frame_times) / sum(self.frame_times)
                performance_stats = {
                    'average_fps': avg_fps,
                    'dropped_frames': self.dropped_frames,
                    'total_frames': self.total_frames,
                    'average_processing_time': sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
                }
                
                if self.data_dir:
                    stats_file = os.path.join(self.data_dir, 'performance_metrics.json')
                    with open(stats_file, 'w') as f:
                        json.dump(performance_stats, f, indent=4)
                    
                    logger.info(f"Performance metrics saved to {stats_file}")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
        finally:
            self.running = False
            logger.info("PyEyeTrack cleanup completed")

    def switch_camera(self, camera_id):
        """Switch to a different camera"""
        if self.cap is not None:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            if self.current_camera is not None:
                self.cap = cv2.VideoCapture(self.current_camera)
            return False
        
        self.current_camera = camera_id
        return True

    def start_recording(self):
        """Start recording"""
        if not self.recording:
            self.recording = True
            self.start_time = time.time()
            print("\nRecording Started")

    def stop_recording(self):
        """Stop recording and save data"""
        if self.recording:
            self.save_eye_data()
            self.recording = False
            print("\nRecording Stopped")

    def pyEyeTrack_runner(
        self,
        source=0,
        pupilTracking=True,
        videoRecording=False,
        audioRecording=False,
        destinationPath="./Output"
    ):
        """
        Main runner method for eye tracking.
        """
        try:
            # Create session directory
            self.session_dir = self.create_session_directory(destinationPath)

            # Initialize video capture
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise Exception("Could not open video source")

            # Initialize window
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

            # Initialize pupil tracker
            if pupilTracking:
                pupil_tracker = PupilTracking()

            print("\nControls:")
            print("Press 'r' to start/stop recording")
            print("Press 'q' to quit")
            print("Press 's' to add a marker/note")
            print("Press 'p' to pause/resume\n")

            while self.running:
                if not self.paused:
                    ret, frame = cap.read()
                    if not ret:
                        self.dropped_frames += 1
                        continue

                    self.total_frames += 1

                    if pupilTracking:
                        frame, eye_coords = pupil_tracker.detect_pupil(frame)
                        
                        if self.recording and eye_coords:
                            timestamp = datetime.now()
                            self.eye_data['timestamp'].append(timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'))
                            self.eye_data['frame_number'].append(self.frame_count)
                            self.eye_data['left_eye_x'].append(eye_coords[0][0])
                            self.eye_data['left_eye_y'].append(eye_coords[0][1])
                            self.eye_data['right_eye_x'].append(eye_coords[1][0])
                            self.eye_data['right_eye_y'].append(eye_coords[1][1])
                            # Add placeholder values for new metrics
                            self.eye_data['left_pupil_size'].append(0)
                            self.eye_data['right_pupil_size'].append(0)
                            self.eye_data['left_blink'].append(0)
                            self.eye_data['right_blink'].append(0)
                            self.eye_data['gaze_x'].append(0)
                            self.eye_data['gaze_y'].append(0)
                            self.eye_data['head_pose_x'].append(0)
                            self.eye_data['head_pose_y'].append(0)
                            self.eye_data['head_pose_z'].append(0)
                            self.eye_data['marker'].append(None)
                            self.frame_count += 1

                    # Add status indicators
                    status_text = []
                    if self.recording:
                        status_text.append("Recording")
                    if self.paused:
                        status_text.append("Paused")
                    if status_text:
                        cv2.putText(frame, " | ".join(status_text), (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Show the frame
                    cv2.imshow(self.window_name, frame)

                # Process key events
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Only process valid key presses
                    if self.process_key(key):
                        if not self.running:
                            break

        except Exception as e:
            print(f"Error in eye tracking: {str(e)}")
        finally:
            if 'cap' in locals():
                cap.release()
            self.cleanup()

                                                    
