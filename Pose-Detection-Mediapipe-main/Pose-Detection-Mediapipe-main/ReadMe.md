# How it's works
1. First import the library that we need
````
import cv2
import mediapipe as mp
````
2. Make the program to connect to the webcam
```
import cv2
import numpy as numpy

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    cv2.imshow('Pose Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
`````
3. Load the module of pose and drawing_utils
````
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
````
4. Determine the minimum percentage
````
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
````
5. Change BGR to RGB
````
image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
````
6. To optimize the program change writeable to False
````
image.flags.writeable = False
````
7. processing
````
results = pose.process(image)
````
8. Change RGB to BGR 
````
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
````
9. Draw the landmark 
````
mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
````                                        