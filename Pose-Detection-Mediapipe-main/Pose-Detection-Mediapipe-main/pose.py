import cv2
import mediapipe as mp
import imutils


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture("sholat.mp4")

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    while True:
        _,frame = cap.read()
        frame = imutils.resize(frame, width= 650,  height= 650)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = pose.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Pose Detection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()






# import cv2
# import mediapipe as mp
# import imutils

# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# cap = cv2.VideoCapture("jumping.mp4")
# with mp_pose.Pose(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         _, frame = cap.read()
#         frame = imutils.resize(frame, width=650, height=650)
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         image.flags.writeable = False
#         results = pose.process(image)

#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         mp_drawing.draw_landmarks(
#             image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#         cv2.imshow('Pose detection', image)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()
