#!/usr/bin/env python3
import os
import sys
import subprocess
from importlib.metadata import version, PackageNotFoundError
import time
from datetime import datetime
import json
import cv2
from admin_gui import launch_admin_gui

def get_default_camera():
    """Get the default camera device"""
    # Try to find an external webcam first
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Get camera info
            if sys.platform == "darwin":  # macOS
                # On macOS, external cameras typically have higher indices
                if i > 0:
                    return i
            else:  # Windows/Linux
                # Try to get camera name
                name = cap.getBackendName()
                if "USB" in name.upper():
                    return i
            cap.release()
    
    # If no external webcam found, use the first available camera
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.release()
        return 0
    
    raise Exception("No camera devices found")

def check_camera_permissions():
    """Check and request camera permissions on macOS"""
    if sys.platform == 'darwin':
        try:
            import AVFoundation
            auth_status = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_('AVMediaTypeVideo')
            if auth_status == AVFoundation.AVAuthorizationStatusNotDetermined:
                AVFoundation.AVCaptureDevice.requestAccessForMediaType_('AVMediaTypeVideo', None)
        except ImportError:
            # If we can't import AVFoundation, we'll rely on OpenCV's permission request
            pass
        except Exception as e:
            print(f"Note: Could not check camera permissions: {e}")
            print("You may need to grant camera permissions manually in System Settings.")

def check_dependencies():
    """Check and install required dependencies"""
    try:
        # Check PyEyeTrack version
        try:
            pyeyetrack_version = version('PyEyeTrack')
            print(f"PyEyeTrack version: {pyeyetrack_version}")
        except PackageNotFoundError:
            print("PyEyeTrack not found. Please install it first.")
            sys.exit(1)
            
        # Check other dependencies
        required = {
            'opencv-python': '4.7.0',
            'numpy': '1.19.5',
            'pandas': '1.2.4',
            'PyQt5': '5.6.0'
        }
        
        for package, min_version in required.items():
            try:
                pkg_version = version(package)
                print(f"{package} version: {pkg_version}")
            except PackageNotFoundError:
                print(f"{package} not found. Installing...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}>={min_version}"])
                
    except Exception as e:
        print(f"Error checking dependencies: {e}")
        sys.exit(1)

def check_model_file():
    """Check if the shape predictor model file exists"""
    model_file = 'shape_predictor_68_face_landmarks.dat'
    if not os.path.exists(model_file):
        print(f"Warning: {model_file} not found.")
        print("Please ensure the model file is in the correct location.")
        return False
    return True

def get_session_metadata():
    """Collect detailed session metadata"""
    metadata = {
        'participant': {
            'id': input("Please enter participant ID: ").strip(),
            'age': input("Participant age (optional, press Enter to skip): ").strip() or None,
            'gender': input("Participant gender (optional, press Enter to skip): ").strip() or None,
            'vision_correction': input("Vision correction? (glasses/contacts/none): ").strip().lower() or 'none'
        },
        'session': {
            'id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'lighting_condition': input("Lighting condition (bright/dim/dark): ").strip().lower() or 'bright',
            'distance_from_screen': input("Approximate distance from screen in cm: ").strip() or '60',
            'screen_resolution': input("Screen resolution (e.g., 1920x1080): ").strip() or '1920x1080'
        },
        'experiment': {
            'task_type': input("Task type (free-viewing/reading/tracking): ").strip().lower() or 'free-viewing',
            'duration': input("Planned duration in minutes: ").strip() or '5',
            'notes': input("Additional notes (optional): ").strip() or None
        }
    }
    return metadata

def validate_metadata(metadata):
    """Validate required metadata fields"""
    if not metadata['participant']['id']:
        raise ValueError("Participant ID cannot be empty")
    
    # Convert numeric fields
    try:
        if metadata['participant']['age']:
            metadata['participant']['age'] = int(metadata['participant']['age'])
        metadata['session']['distance_from_screen'] = float(metadata['session']['distance_from_screen'])
        metadata['experiment']['duration'] = float(metadata['experiment']['duration'])
    except ValueError as e:
        raise ValueError(f"Invalid numeric value in metadata: {str(e)}")

def setup_session_directory(metadata):
    """Create and setup session directory structure"""
    base_dir = "Output"
    session_dir = os.path.join(
        base_dir,
        f"participant_{metadata['participant']['id']}",
        f"session_{metadata['session']['id']}"
    )
    
    # Create directory structure
    subdirs = ['data', 'metadata', 'analysis']
    for subdir in subdirs:
        os.makedirs(os.path.join(session_dir, subdir), exist_ok=True)
    
    # Save metadata
    metadata_file = os.path.join(session_dir, 'metadata', 'session_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return session_dir

def main():
    """Main function to run the eye tracking application"""
    print("\n=== PyEyeTrack Application ===\n")
    
    try:
        # Check dependencies and permissions
        check_dependencies()
        check_camera_permissions()
        
        if not check_model_file():
            print("Warning: Proceeding without model file. Eye tracking may not work correctly.")
        
        # Get default camera
        try:
            default_camera = get_default_camera()
            print(f"Using camera device {default_camera}")
        except Exception as e:
            print(f"Error finding camera: {e}")
            sys.exit(1)
        
        # Collect and validate session metadata
        metadata = get_session_metadata()
        validate_metadata(metadata)
        
        # Setup session directory
        session_dir = setup_session_directory(metadata)
        
        print("\nSession Information:")
        print(f"Participant ID: {metadata['participant']['id']}")
        print(f"Session ID: {metadata['session']['id']}")
        print(f"Task Type: {metadata['experiment']['task_type']}")
        print(f"Duration: {metadata['experiment']['duration']} minutes")
        
        # Import and initialize the eye tracker
        from pyEyeTrack import PyEyeTrackRunner
        tracker = PyEyeTrackRunner(
            participant_id=metadata['participant']['id'],
            session_id=metadata['session']['id'],
            metadata=metadata
        )
        
        # Launch admin GUI
        app, admin_window = launch_admin_gui(tracker)
        
        print("\nStarting eye tracking...")
        print("Controls:")
        print("Press 'r' to start/stop recording")
        print("Press 'q' to quit")
        print("Press 's' to add a marker/note")
        print("Press 'p' to pause/resume\n")
        
        # Initialize camera
        tracker.switch_camera(default_camera)
        
        # Run the eye tracking
        tracker.pyEyeTrack_runner(
            source=default_camera,  # Use detected camera
            pupilTracking=True,  # Enable pupil tracking
            videoRecording=False,  # Disable video recording
            audioRecording=False,  # Disable audio recording
            destinationPath=session_dir  # Use organized session directory
        )
        
        # Start Qt event loop
        sys.exit(app.exec_())
        
    except KeyboardInterrupt:
        print("\nEye tracking stopped by user.")
    except Exception as e:
        print(f"Error running PyEyeTrack: {e}")
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 