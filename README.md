# Advanced-Face-Recognition-System-Project

A real-time face recognition system built using Python, OpenCV, and the face_recognition library. This project allows you to register new users by capturing their facial features and later recognize them live using your system's webcam.

Features
--------
- Register new faces with a unique ID and name.
- Store facial encodings persistently using `pickle`.
- Live face detection and recognition via webcam.
- Real-time video frame processing.
- Option to label recognized faces or show as "Unknown".

Requirements
------------
Install the required Python libraries:

    pip install opencv-python face_recognition numpy

You may also need to install `dlib` (used internally by face_recognition). On Windows, this can sometimes require Visual Studio Build Tools.

How to Use
----------

1. Register a New User

Run main.py to register a new face:

    python main.py

- You'll be prompted to enter a name and a unique ID.
- The system will open the webcam.
- Press 's' to capture a face image. This will be done 5 times for better accuracy.
- Press 'q' anytime to quit.

Captured data is stored in:
- ref_name.pkl – maps user ID to name.
- ref_embed.pkl – stores multiple face encodings per user.

2. Run Face Recognition

Start the recognition script:

    python recognition.py

- The webcam opens and begins scanning for faces.
- If a registered face is recognized, it is labeled with the user’s name.
- Unknown faces are labeled as "Unknown".
- Press 'q' to quit the program.

How It Works
------------
- Facial embeddings are generated using the face_recognition library.
- Known face embeddings are compared to new camera frames using cosine similarity.
- A match is declared if a similarity threshold is met.
- Names are drawn using OpenCV’s GUI features.

Technologies Used
-----------------
- Python 3
- OpenCV – for webcam interaction and drawing
- face_recognition – for facial embedding and comparison
- pickle – to store user data locally
Notes
-----
- Ensure your webcam is working.
- Make sure lighting conditions are reasonable when capturing or recognizing.
- Data files (.pkl) will persist across runs, so re-registration is not necessary.

