Notebook and functions used to try various classification tasks on processed multimodal vocal data.

1. Start a virtual environment in python 3.10
conda create -n pose_face_test_env python=3.10
conda activate pose_face_test_env

2. Install requirements: 
conda install -c conda-forge cmake (DO THIS FIRST)
pip install -r requirements.txt

3. Get the dlib model from
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

4. Running the code:
Follow the step by step process outlines in the python notebook. If interested in the internal workings of the
functions used, look at the other python scripts containing them.

It is recommended that you install an extension to view videos within VSCode to ease the workflow.

5. RUNNING THE SCRIPTS
If modules are not recognised, make sure that your project is in your path, 
export PYTHONPATH="path_to/feature_based_classifiers:$PYTHONPATH"
e.g. I would run the following:
export PYTHONPATH="/Users/Suvi/Documents/UPF/THESIS/feature_based_classifiers:$PYTHONPATH"

6. Run MediaPipe live face and pose estimation by simply running the python scripts. 
    python scripts/mediapipe_face_live.py 
    python scripts/mediapipe_pose_live.py  

7. 

