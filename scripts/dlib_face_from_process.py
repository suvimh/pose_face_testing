import cv2
import dlib
import pandas as pd
PROTO_PATH = "data/shape_predictor_68_face_landmarks.dat"


def get_dlib_face_landmarks(input_video_path, output_video_path=None, output_video=False):
    """
    Extracts facial landmarks from a video using dlib.

    Args:
        input_video_path (str): The path to the input video file.
        output_video_path (str, optional): The path to save the output video file with facial landmarks drawn. Defaults to None.
        output_video (bool, optional): Whether to save the output video file. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing facial landmarks for each frame.
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PROTO_PATH)  # Path to the dlib model

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return pd.DataFrame()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if output_video:
        out = cv2.VideoWriter(output_video_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

    face_landmarks_list = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Finished processing or error reading frame at frame count: {frame_count}")
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            face_landmarks_list.append([None] * 68)  # Assuming 68 landmarks for consistency
            
            if output_video:
                # Draw a red circle in the top-left corner to indicate no face detected
                cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)
                out.write(frame)
        else:
            for face in faces:
                landmarks = predictor(gray, face)
                face_landmarks = [[p.x, p.y, None] for p in landmarks.parts()]
                face_landmarks_list.append(face_landmarks)

                if output_video:
                    for p in landmarks.parts():
                        cv2.circle(frame, (p.x, p.y), 1, (0, 255, 0), -1)
                    out.write(frame)

    cap.release()
    if output_video:
        out.release()
    cv2.destroyAllWindows()

    return pd.DataFrame(face_landmarks_list)
