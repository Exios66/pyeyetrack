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
        Validate eye tracking data quality with proper null value handling
        Args:
            eye_data: Dictionary containing eye tracking data
        Returns:
            float: Quality score between 0 and 1
        """
        if eye_data is None:
            return 0.0
            
        if not isinstance(eye_data, dict):
            return 0.0
            
        quality_score = 1.0
        deductions = []
        
        # Check pupil size if available and not None
        left_pupil = eye_data.get('left_pupil_size')
        if left_pupil is not None and isinstance(left_pupil, (int, float)):
            if not (self.validation_ranges['pupil_size'][0] <= left_pupil <= self.validation_ranges['pupil_size'][1]):
                quality_score -= 0.2
                deductions.append('pupil_size_out_of_range')
        
        # Check gaze coordinates if available and not None
        gaze_x = eye_data.get('gaze_x')
        gaze_y = eye_data.get('gaze_y')
        if gaze_x is not None and gaze_y is not None and isinstance(gaze_x, (int, float)) and isinstance(gaze_y, (int, float)):
            if not (0 <= gaze_x <= self.validation_ranges['gaze_x'][1]):
                quality_score -= 0.2
                deductions.append('gaze_x_out_of_range')
            if not (0 <= gaze_y <= self.validation_ranges['gaze_y'][1]):
                quality_score -= 0.2
                deductions.append('gaze_y_out_of_range')
        
        # Check head pose if available and not None
        for pose in ['head_pose_x', 'head_pose_y', 'head_pose_z']:
            pose_val = eye_data.get(pose)
            if pose_val is not None and isinstance(pose_val, (int, float)):
                if not (self.validation_ranges['head_pose'][0] <= pose_val <= self.validation_ranges['head_pose'][1]):
                    quality_score -= 0.1
                    deductions.append(f'{pose}_out_of_range')
        
        # Log quality issues if any
        if deductions:
            logger.debug(f"Data quality issues detected: {', '.join(deductions)}")
        
        return max(0.0, quality_score)

    def create_session_directory(self, base_dir):
        """Create a new session directory with timestamp and proper organization."""
        # Determine if this is a pilot or live session from metadata
        session_type = self.metadata.get('session_type', 'live')  # Default to live if not specified
        data_type_dir = "pilot_data" if session_type == 'pilot' else "live_data"
        
        # Create the full directory path with pilot/live separation
        self.session_dir = os.path.join(
            base_dir,
            data_type_dir,
            f"participant_{self.participant_id}",
            f"session_{self.session_id}"
        )
        
        # Set up data directories
        self.data_dir = os.path.join(self.session_dir, "data", "raw_data")
        self.processed_dir = os.path.join(self.session_dir, "data", "processed_data")
        self.export_dir = os.path.join(self.session_dir, "exports")
        self.log_dir = os.path.join(self.session_dir, "logs")
        
        # Create all necessary directories
        for directory in [self.data_dir, self.processed_dir, self.export_dir, self.log_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize session log
        log_file = os.path.join(self.log_dir, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        with open(log_file, 'w') as f:
            f.write(f"Session Type: {session_type}\n")
            f.write(f"Participant ID: {self.participant_id}\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logger.info(f"Created session directory structure in {self.session_dir}")
        logger.info(f"Session Type: {session_type}")
        logger.info(f"Raw data will be saved to: {self.data_dir}")
        
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

    def save_eye_data(self) -> None:
        """
        Save eye tracking data to CSV file with enhanced organization and validation.
        
        Raises:
            Exception: If data directory is not set
            ValueError: If eye data is invalid or empty
            OSError: If there's an error creating directories or saving files
        """
        if self.data_dir is None:
            raise Exception("Data directory not set")
        
        if not self.eye_data or len(self.eye_data.get('timestamp', [])) == 0:
            raise ValueError("No eye tracking data available to save")
        
        try:
            # Generate timestamp for unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create DataFrame with all collected metrics
            df = pd.DataFrame(self.eye_data)
            
            # Calculate recording statistics
            stats = self._calculate_recording_stats(timestamp)
            
            # Ensure raw data directory exists
            raw_data_dir = os.path.join(self.data_dir, "raw_data")
            os.makedirs(raw_data_dir, exist_ok=True)
            
            # Save raw data with proper naming convention
            csv_filename = f"eye_tracking_data_{self.participant_id}_{self.session_id}_{timestamp}.csv"
            csv_path = os.path.join(raw_data_dir, csv_filename)
            df.to_csv(csv_path, index=False)
            os.chmod(csv_path, 0o644)  # rw-r--r--
            
            # Save additional files
            self._save_additional_files(self.session_dir, timestamp, stats)
            
            # Log recording summary
            self._log_recording_summary(stats)
            
            # Clear data after successful save
            self.eye_data = {k: [] for k in self.eye_data.keys()}
            logger.info(f"Data successfully saved to {csv_path}")
            
        except Exception as e:
            logger.error(f"Error saving eye tracking data: {str(e)}")
            raise

    def _calculate_recording_stats(self, timestamp: str) -> Dict[str, Any]:
        """Calculate comprehensive recording statistics with proper null value handling."""
        recording_duration = time.time() - self.start_time if self.start_time else 0
        frame_rate = self.total_frames / recording_duration if recording_duration > 0 else 0
        
        # Helper function to safely count valid gaze points
        def count_valid_gaze_points():
            count = 0
            for x, y in zip(self.eye_data['gaze_x'], self.eye_data['gaze_y']):
                if x is not None and y is not None and isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    if x > 0 and y > 0:
                        count += 1
            return count
        
        # Helper function to safely calculate mean of data quality scores
        def calculate_mean_quality():
            quality_scores = [score for score in self.eye_data['data_quality'] if score is not None]
            return np.mean(quality_scores) if quality_scores else 0
        
        # Helper function to safely count blinks
        def count_total_blinks():
            left_blinks = sum(1 for b in self.eye_data['left_blink'] if b is not None and b > 0)
            right_blinks = sum(1 for b in self.eye_data['right_blink'] if b is not None and b > 0)
            return left_blinks + right_blinks
        
        return {
            'session_info': {
                'participant_id': self.participant_id,
                'session_id': self.session_id,
                'session_type': self.metadata.get('session_type', 'live'),
                'timestamp': timestamp,
                'recording_start': self.start_time,
                'recording_end': time.time()
            },
            'performance': {
                'total_frames': self.total_frames,
                'dropped_frames': self.dropped_frames,
                'frame_rate': frame_rate,
                'recording_duration': recording_duration,
                'camera_id': self.current_camera
            },
            'data_quality': {
                'average_quality_score': calculate_mean_quality(),
                'total_blinks': count_total_blinks(),
                'valid_gaze_points': count_valid_gaze_points(),
                'data_completeness': len([x for x in self.eye_data['timestamp'] if x is not None]) / self.total_frames if self.total_frames > 0 else 0
            }
        }

    def _save_additional_files(self, session_dir: str, timestamp: str, stats: Dict[str, Any]) -> None:
        """Save additional files like markers, notes, and statistics."""
        # Save markers if available
        if self.markers:
            markers_file = os.path.join(session_dir, "metadata", f"markers_{timestamp}.json")
            with open(markers_file, 'w') as f:
                json.dump(self.markers, f, indent=4)
            os.chmod(markers_file, 0o644)
        
        # Save notes if available
        if self.notes:
            notes_file = os.path.join(session_dir, "metadata", f"notes_{timestamp}.json")
            with open(notes_file, 'w') as f:
                json.dump(self.notes, f, indent=4)
            os.chmod(notes_file, 0o644)
        
        # Save statistics
        stats_file = os.path.join(session_dir, "metadata", f"recording_stats_{timestamp}.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=4)
        os.chmod(stats_file, 0o644)

    def _log_recording_summary(self, stats: Dict[str, Any]) -> None:
        """Log a summary of the recording session."""
        logger.info("\nRecording Summary:")
        logger.info(f"Participant ID: {self.participant_id}")
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Total Frames: {stats['performance']['total_frames']}")
        logger.info(f"Dropped Frames: {stats['performance']['dropped_frames']}")
        logger.info(f"Average Frame Rate: {stats['performance']['frame_rate']:.2f} fps")
        logger.info(f"Recording Duration: {stats['performance']['recording_duration']:.2f} seconds")
        logger.info(f"Data Quality Score: {stats['data_quality']['average_quality_score']:.2f}")

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

    def switch_camera(self, camera_id: int) -> bool:
        """
        Switch to a different camera
        Args:
            camera_id: ID of the camera to switch to
        Returns:
            bool: True if switch successful, False otherwise
        Raises:
            ValueError: If camera_id is invalid
        """
        if camera_id < 0:
            raise ValueError(f"Invalid camera ID: {camera_id}")
            
        if self.cap is not None:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            logger.error(f"Could not open camera {camera_id}")
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
        Main runner method for eye tracking
        Args:
            source: Camera source (default: 0)
            pupilTracking: Enable pupil tracking (default: True)
            videoRecording: Enable video recording (default: False)
            audioRecording: Enable audio recording (default: False)
            destinationPath: Output directory path (default: "./Output")
        """
        try:
            # Create session directory first
            if not os.path.exists(destinationPath):
                os.makedirs(destinationPath)
            
            # Create session directory structure
            self.create_session_directory(destinationPath)
            
            # Initialize camera with proper detection
            logger.info("Detecting available cameras...")
            if not self.initialize_camera(source):
                raise Exception("Could not initialize any camera. Please check your camera connection.")
            
            # Initialize trackers
            if pupilTracking:
                try:
                    pupil_tracker = PupilTracking(source=source, session_id=self.session_id)
                    logger.info("PupilTracking initialized")
                except Exception as e:
                    logger.error(f"Error initializing pupil tracker: {str(e)}")
                    raise
            
            # Initialize recorders if needed
            video_recorder = None
            audio_recorder = None
            if videoRecording:
                video_recorder = VideoRecorder(self.cap, os.path.join(self.data_dir, "video_recording.avi"))
            if audioRecording and AUDIO_AVAILABLE:
                audio_recorder = AudioRecorder(os.path.join(self.data_dir, "audio_recording.wav"))
            
            # Initialize data collection
            self.frame_count = 0
            self.total_frames = 0
            self.eye_data = {k: [] for k in self.eye_data.keys()}  # Reset data storage
            
            # Create window with proper settings
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            
            while self.running:
                if not self.paused:
                    # Read frame with error handling
                    try:
                        ret, frame = self.cap.read()
                        if not ret or frame is None:
                            self.dropped_frames += 1
                            logger.warning("Failed to read frame from camera")
                            # Try to reinitialize camera if multiple frames are dropped
                            if self.dropped_frames > 10:
                                logger.warning("Too many dropped frames, attempting to reinitialize camera...")
                                if not self.initialize_camera(source):
                                    raise Exception("Failed to reinitialize camera")
                                self.dropped_frames = 0
                            continue
                    except Exception as e:
                        logger.error(f"Error reading frame: {str(e)}")
                        continue

                    self.total_frames += 1
                    current_time = datetime.now()

                    try:
                        if pupilTracking:
                            frame, eye_coords = pupil_tracker.detect_pupil(frame)
                            
                            # Always collect data when recording, even if eye tracking fails
                            if self.recording:
                                self.eye_data['timestamp'].append(current_time.strftime('%Y-%m-%d %H:%M:%S.%f'))
                                self.eye_data['frame_number'].append(self.frame_count)
                                
                                if eye_coords and len(eye_coords) == 2:
                                    self.eye_data['left_eye_x'].append(float(eye_coords[0][0]))
                                    self.eye_data['left_eye_y'].append(float(eye_coords[0][1]))
                                    self.eye_data['right_eye_x'].append(float(eye_coords[1][0]))
                                    self.eye_data['right_eye_y'].append(float(eye_coords[1][1]))
                                else:
                                    # If eye tracking fails, store None values
                                    self.eye_data['left_eye_x'].append(None)
                                    self.eye_data['left_eye_y'].append(None)
                                    self.eye_data['right_eye_x'].append(None)
                                    self.eye_data['right_eye_y'].append(None)
                                
                                # Add other metrics with proper type handling
                                self.eye_data['left_pupil_size'].append(None)
                                self.eye_data['right_pupil_size'].append(None)
                                self.eye_data['left_blink'].append(0)
                                self.eye_data['right_blink'].append(0)
                                self.eye_data['gaze_x'].append(None)
                                self.eye_data['gaze_y'].append(None)
                                self.eye_data['head_pose_x'].append(None)
                                self.eye_data['head_pose_y'].append(None)
                                self.eye_data['head_pose_z'].append(None)
                                self.eye_data['marker'].append(None)
                                self.eye_data['data_quality'].append(1.0 if eye_coords else 0.0)
                                
                                self.frame_count += 1
                                
                                # Periodically save data to prevent data loss
                                if self.frame_count % 300 == 0:  # Save every 300 frames
                                    try:
                                        self.save_eye_data()
                                    except Exception as e:
                                        logger.warning(f"Error in periodic data save: {str(e)}")
                    
                    except Exception as e:
                        logger.error(f"Error processing frame: {str(e)}")
                        continue

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
                    try:
                        cv2.imshow(self.window_name, frame)
                    except Exception as e:
                        logger.error(f"Error displaying frame: {str(e)}")

                # Process key events
                try:
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255:  # Only process valid key presses
                        if self.process_key(key):
                            if not self.running:
                                break
                except Exception as e:
                    logger.error(f"Error processing key event: {str(e)}")

        except Exception as e:
            logger.error(f"Error in eye tracking: {str(e)}")
            raise
        finally:
            # Cleanup
            try:
                if self.recording:
                    self.save_eye_data()
                if hasattr(self, 'cap') and self.cap is not None:
                    self.cap.release()
                cv2.destroyAllWindows()
                self.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")

    def detect_cameras(self) -> list:
        """
        Detect available cameras and their indices.
        Returns:
            list: List of available camera indices
        """
        available_cameras = []
        max_to_test = 2  # Limit initial search to first 2 indices to avoid long startup
        
        logger.info("Detecting available cameras...")
        for i in range(max_to_test):
            temp_cap = cv2.VideoCapture(i, cv2.CAP_ANY)
            if temp_cap.isOpened():
                ret, frame = temp_cap.read()
                if ret:
                    available_cameras.append(i)
                    logger.info(f"Camera {i} is available")
                temp_cap.release()
            else:
                logger.debug(f"Camera {i} is not available")
        
        if not available_cameras:
            logger.warning("No cameras detected!")
        
        return available_cameras

    def initialize_camera(self, source=0) -> bool:
        """
        Initialize the camera with proper error handling.
        Args:
            source: Camera index to try
        Returns:
            bool: True if camera was successfully initialized
        """
        available_cameras = self.detect_cameras()
        
        if not available_cameras:
            logger.error("No cameras available. Please check your camera connection.")
            return False
            
        if source not in available_cameras:
            logger.warning(f"Requested camera {source} not available. Using camera {available_cameras[0]} instead.")
            source = available_cameras[0]
        
        try:
            self.cap = cv2.VideoCapture(source, cv2.CAP_ANY)
            if not self.cap.isOpened():
                raise Exception(f"Failed to open camera {source}")
                
            # Try to read a test frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise Exception(f"Failed to read from camera {source}")
                
            self.current_camera = source
            logger.info(f"Successfully initialized camera {source}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {str(e)}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False

                                                    
