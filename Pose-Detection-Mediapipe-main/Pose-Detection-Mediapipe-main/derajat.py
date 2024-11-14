import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def menghitung_derajat(pertama,kedua,ketiga):
    pertama = np.array(pertama) 
    kedua = np.array(kedua) 
    ketiga = np.array(ketiga) 
    
    radians = np.arctan2(ketiga[1]-kedua[1], ketiga[0]-kedua[0]) - np.arctan2(pertama[1]-kedua[1], pertama[0]-kedua[0])
    derajat = np.abs(radians*180.0/np.pi)
    
    if derajat >180.0: # dapat disesuaikan dengan maksimal pergerakan dari anggota badan tersebut
        derajat = 360-derajat
        
    return derajat


cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w, c = frame.shape
# Initiate holistic model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        #print(format(frame.shape))
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = pose.process(image)
        # print(results.face_landmarks)
        
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        edge = np.zeros((512,512,3), np.uint8)
        
        # 1. Draw face landmarks

        # 4. Pose Detections
        try:
            landmarks = results.pose_landmarks.landmark
            bahu_kanan = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y] # hanya memerlukan koordinat x dan y 
            siku_kanan = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            pergelangan_kanan = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            result_derajat = menghitung_derajat(bahu_kanan, siku_kanan, pergelangan_kanan)
           
            cv2.putText(image, str(result_derajat), 
                           tuple(np.multiply(siku_kanan,[600,600]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
        except:
            pass
        mp_drawing.draw_landmarks(edge, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        cv2.imshow('Raw Webcam Feed', image)
        cv2.imshow('Raw', edge)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
