'''
Testing whether other face mesh libraries are robust against issues in the dataset 
- participants turning their head
- camera distance is quite far from the particpant

mediapipe works when the face is presented close and then moved further, 
but if the head turns when far further from the camera, it cannot reregister the face points

REFERENCES:
https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb
https://developers.google.com/mediapipe/solutions/vision/face_landmarker 

https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
https://www.analyticsvidhya.com/blog/2021/10/face-mesh-application-using-opencv-and-dlib/

https://pyimagesearch.com/2021/04/12/opencv-haar-cascades/
https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html 

https://github.com/italojs/facial-landmarks-recognition/tree/master
http://dlib.net/face_landmark_detection.py.html

'''

import cv2
import mediapipe as mp
import os
import re
import dlib

PROTO_PATH = "data/shape_predictor_68_face_landmarks.dat"


def test_mediapipe(file):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(script_dir)
    video_path = os.path.join(parent_dir, file)

    output_video_name = re.sub(r'(?i)\.mp4$', '', video_path) + '_MEDIAPIPE_landmarks.mp4'

    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    # For input video:
    cap = cv2.VideoCapture(video_path)

    # For output video:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_name, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

    face_mesh_landmarks_list = []

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty video frame.")
                break

            image_copy = image.copy()
            image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

            # Perform face mesh estimation
            results = face_mesh.process(image_rgb)
            if results.multi_face_landmarks:
                landmarks_list = []
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = []
                    for landmark in face_landmarks.landmark:
                        landmarks.append([landmark.x, landmark.y, landmark.z if landmark.HasField('z') else None])
                    landmarks_list.append(landmarks)

                    mp_drawing.draw_landmarks(
                        image=image_copy,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=image_copy,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=image_copy,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())

                face_mesh_landmarks_list.append(landmarks_list)

            # Write the frame to the output video
            out.write(image_copy)

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

def test_open_cv(file):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Define the path to the shape predictor model file and to video file
    video_path = os.path.join(script_dir, "..", file)

    output_video_name = re.sub(r'(?i)\.mp4$', '', video_path) + '_OPEN_CV_landmarks.mp4'

    # Load pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    video_capture = cv2.VideoCapture(video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec
    out = cv2.VideoWriter(output_video_name, fourcc, 30, (frame_width, frame_height))

    # Process the video frames
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw face mesh
            for i in range(x, x + w):
                for j in range(y, y + h):
                    cv2.circle(frame, (i, j), 1, (0, 255, 0), -1)

        out.write(frame)

    video_capture.release()
    out.release()

# def test_dlib(file, video_path=None):
#     '''
#     This example program shows how to find frontal human faces in an image and
#     estimate their pose.  The pose takes the form of 68 landmarks.  These are
#     points on the face such as the corners of the mouth, along the eyebrows, on
#     the eyes, and so forth.

#     The face detector we use is made using the classic Histogram of Oriented
#     Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#     and sliding window detection scheme.  The pose estimator was created by
#     using dlib's implementation of the paper:
#         One Millisecond Face Alignment with an Ensemble of Regression Trees by
#         Vahid Kazemi and Josephine Sullivan, CVPR 2014
#     and was trained on the iBUG 300-W face landmark dataset (see
#     https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):  
#         C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 
#         300 faces In-the-wild challenge: Database and results. 
#         Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
#     You can get the trained model file from:
#     http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.
#     Note that the license for the iBUG 300-W dataset excludes commercial use.
#     So you should contact Imperial College London to find out if it's OK for
#     you to use this model file in a commercial product.


#     Also, note that you can train your own models using dlib's machine learning
#     tools. See train_shape_predictor.py to see an example.
#     '''
#     # Initialize dlib's face detector and facial landmark predictor
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor(PROTO_PATH)  # Path to the dlib model

#     cap = cv2.VideoCapture(file)
#     if not cap.isOpened():
#         print(f"Error: Could not open video {file}")
#         return

#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Generate output video path
#     output_video_name = re.sub(r'(?i)\.mp4$', '', os.path.basename(file)) + '_MEDIAPIPE_landmarks.mp4'
#     output_video_path = os.path.join(os.path.dirname(file), output_video_name)

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height))

#     frame_count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector(gray)

#         for face in faces:
#             landmarks = predictor(gray, face)
#             for p in landmarks.parts():
#                 cv2.circle(frame, (p.x, p.y), 1, (0, 255, 0), -1)

#         out.write(frame)
#         frame_count += 1

#     print(f"Processed {frame_count} frames")

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

def test_dlib(file, output_video_path=None):
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

    script_dir = os.path.dirname(os.path.realpath(__file__))
    video_path = os.path.join(script_dir, "..", file)
    output_video_path = re.sub(r'(?i)\.mp4$', '', video_path) + '_DLIB_face_landmarks.mp4'

    cap = cv2.VideoCapture(file)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_video_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            face_landmarks = [[p.x, p.y, None] for p in landmarks.parts()]
            
            for p in landmarks.parts():
                cv2.circle(frame, (p.x, p.y), 1, (0, 255, 0), -1)
            out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return 

def get_face_pose_output_paths(input_file_path, output_path, landmark_type):
    '''
    Returns the output paths for the video file with landmarks and the JSON file with landmark locations.

    Parameters:
    - input_file_path (str): The path of the input video file.
    - output_path (str): The path where the output files will be saved.
    - landmark_type (str): The type of landmarks to be extracted. Can be either 'face' for dlib face landmarks file or 'pose' for mediapipe pose estimation.

    Returns:
    - output_video_path (str): The path of the output video file with landmarks.
    - output_json_path (str): The path of the output JSON file with landmark locations.
    '''
    input_filename = os.path.basename(input_file_path)

    output_video_name = re.sub(r'(?i)\.mp4$', '', input_filename) + f'_with_{landmark_type}_landmarks.mp4'
    output_video_path = os.path.join(output_path, output_video_name)
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    output_json_name = re.sub(r'(?i)\.mp4$', '', input_filename) + f'_{landmark_type}_landmark_locations.json'
    output_json_path = os.path.join(output_path, output_json_name)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    return output_video_path, output_json_path


def test_open_cv_with_dlib(file, protoPath=PROTO_PATH, video_path=None):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    if video_path is None:
        video_path = os.path.join(script_dir, "..", file)

    output_video_name = re.sub(r'(?i)\.mp4$', '', video_path) + '_OPEN_CV_DLIB_landmarks.mp4'

    # Load pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load pre-trained facial landmark detection model
    predictor = dlib.shape_predictor(protoPath)

    video_capture = cv2.VideoCapture(video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec
    out = cv2.VideoWriter(output_video_name, fourcc, 30, (frame_width, frame_height))

    # Process the video frames
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Detect facial landmarks
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            landmarks = predictor(gray, rect)

            # Draw face mesh
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        out.write(frame)

    video_capture.release()
    out.release()