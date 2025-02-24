# PyEyeTrack Application

A Python-based eye tracking application that uses OpenCV for face detection and pupil tracking.

## Features

- Real-time pupil tracking
- Session-based data collection
- CSV output with timestamps
- Visual feedback with eye region highlighting
- Automatic dependency management

## Requirements

- Python 3.6 or higher
- Webcam
- Good lighting conditions

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/yourusername/pyeyetrack.git
cd pyeyetrack
```

2. Launch the application:

```bash
python launch_app.py
```

The launch script will:

- Check and install required dependencies
- Create necessary directories
- Download required model files (if needed)
- Guide you through session setup
- Start the eye tracking application

## Usage

1. When you run `launch_app.py`, you'll be prompted to:
   - Enter a Participant ID
   - The system will automatically generate a unique session ID using the participant ID and timestamp

2. During the tracking session:
   - Position yourself in front of the camera
   - Ensure good lighting conditions
   - The application will show:
     - Green rectangles around detected eyes
     - Red dots for pupil centers
     - Session ID display

3. Controls:
   - Press 'ESC' to stop tracking
   - The application will automatically save data to the Output directory

## Output

The application generates CSV files in the `Output` directory with the following format:

- Filename: `participant_<participant_id>_<timestamp>.csv`
- Contents:
  - Session_ID: Unique session identifier
  - Time: Timestamp for each measurement
  - Left_Pupil_X: X-coordinate of left pupil
  - Left_Pupil_Y: Y-coordinate of left pupil
  - Right_Pupil_X: X-coordinate of right pupil
  - Right_Pupil_Y: Y-coordinate of right pupil

## Troubleshooting

1. If you encounter webcam access issues:
   - Make sure no other application is using the webcam
   - Check webcam permissions

2. If pupil detection is unreliable:
   - Improve lighting conditions
   - Adjust your position relative to the camera
   - Ensure your eyes are clearly visible

3. If dependencies fail to install:
   - Try running `pip install -r requirements.txt` manually
   - Check your Python version and environment

## License

This project is licensed under the MIT License - see the LICENSE file for details.
