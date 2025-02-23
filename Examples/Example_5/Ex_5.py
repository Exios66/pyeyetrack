from pyEyeTrack.PyEyeTrackRunnerClass import pyEyeTrack

def run_eye_tracking_session(session_id):
    """
    Run an eye tracking session with the specified session ID
    Args:
        session_id: Unique identifier for the tracking session
    """
    ptr = pyEyeTrack()
    ptr.pyEyeTrack_runner(
        pupilTracking=True,
        eyeTrackingLog=True,
        eyeTrackingFileName='participant',
        session_id=session_id
    )

if __name__ == "__main__":
    # Get session ID from user
    session_id = input("Enter participant/session ID: ")
    
    # Run eye tracking with the provided session ID
    run_eye_tracking_session(session_id)
    
