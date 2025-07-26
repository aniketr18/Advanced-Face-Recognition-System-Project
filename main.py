import cv2 
import face_recognition
import pickle

name = input("Enter name: ")
ref_id = input("Enter id: ")

try:
    f = open("ref_name.pkl", "rb")
    ref_dict = pickle.load(f)
    f.close()
except FileNotFoundError:
    ref_dict = {}

ref_dict[ref_id] = name

with open("ref_name.pkl", "wb") as f:
    pickle.dump(ref_dict, f)

try:
    with open("ref_embed.pkl", "rb") as f:
        embed_dict = pickle.load(f)
except FileNotFoundError:
    embed_dict = {}

for i in range(5):
    webcam = cv2.VideoCapture(0)
    
    while True:
        check, frame = webcam.read()
        cv2.imshow("Capturing", frame)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        key = cv2.waitKey(1)

        if key == ord('s'): 
            face_locations = face_recognition.face_locations(rgb_small_frame)
            if face_locations:
                face_encoding = face_recognition.face_encodings(frame)[0]
                if ref_id in embed_dict:
                    embed_dict[ref_id].append(face_encoding)
                else:
                    embed_dict[ref_id] = [face_encoding]
                webcam.release()
                cv2.destroyAllWindows()
                break
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            cv2.destroyAllWindows()
            break

with open("ref_embed.pkl", "wb") as f:
    pickle.dump(embed_dict, f)
