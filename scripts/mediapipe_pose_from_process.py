import cv2
import mediapipe as mp
import pandas as pd

def get_mediapipe_landmarks(input_video_path, output_video_path=None, output_video=False):
    """
    Extracts landmarks from a video using the MediaPipe Pose model.

    Args:
        input_video_path (str): The path to the input video file.
        output_video_path (str, optional): The path to save the output video file. Defaults to None.
        output_video (bool, optional): Whether to save the output video. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing landmarks for each frame. Columns are landmark numbers
                     and values are dictionaries with 'x', 'y', and 'z' (if available).
    """
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if output_video:
        out = cv2.VideoWriter(output_video_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    pose_landmarks_list = []

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty video frame.")
                break

            image_copy = image.copy()
            image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

            pose_results = pose.process(image_rgb)
            if pose_results.pose_landmarks:
                landmarks = {}
                for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                    if landmark.visibility < 0.5:
                        # Landmark not visible, set as None
                        landmarks[idx] = None
                    else:
                        landmarks[idx] = {
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z if landmark.HasField('z') else None
                        }
                
                pose_landmarks_list.append(landmarks)

                mp_drawing.draw_landmarks(
                    image_copy,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            if output_video:
                out.write(image_copy)

        cap.release()
        if output_video:
            out.release()
        cv2.destroyAllWindows()

    return pd.DataFrame(pose_landmarks_list)
