# Session-based Eye Tracking Example

## Description
This example demonstrates how to use the session ID functionality to track and save eye tracking data with participant identifiers.

## Features
- Prompts for a participant/session ID at the start
- Tracks pupil position for both eyes
- Saves data to a CSV file with the session ID in the filename
- Displays session ID on the tracking window
- Real-time visualization of eye tracking

## Output
The application generates a CSV file in the `Output` directory with the following format:
- Filename: `participant_<session_id>.csv`
- Contents:
  - Session_ID: The participant identifier
  - Time: Timestamp for each measurement
  - Left_Pupil_X: X-coordinate of left pupil
  - Left_Pupil_Y: Y-coordinate of left pupil
  - Right_Pupil_X: X-coordinate of right pupil
  - Right_Pupil_Y: Y-coordinate of right pupil

## Usage
1. Run the script: `python Ex_5.py`
2. Enter a participant/session ID when prompted
3. The eye tracking window will open
4. Press 'ESC' to stop tracking
5. Data will be saved automatically to the Output directory

## Note
Make sure you have good lighting conditions and are positioned correctly in front of the camera for optimal tracking results.

## Capturing audio and video along with UI
Description: This application/demo records audio and video on user-specified UI. This video can be used later to perform pupil tracking as done in example 4.<br>

Working/ Library Function: This application/demo uses audio recording and video recording functionalities of the library. The application can take any user-specified UI as the input.<br>

Output: The program returns an audio and video file.

Dependency:<br>
To run this example install [PyQt 5.6.0](https://anaconda.org/conda-forge/pyqt/)
 
