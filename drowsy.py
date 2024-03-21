import cv2
import numpy as np
import dlib
from imutils import face_utils

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
sleep = 0
drowsy = 0
active = 0
status = ""
status2 = ""
yawn = 0
ear = "EAR:"
mar = "MAR:"

color = (255, 255, 0)
color2 = (256, 150, 66)


def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist


def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2
    elif ratio > 0.21 and ratio <= 0.25:
        return 1
    else:
        return 0


def yawning(landmarks):
    up1 = compute(landmarks[61], landmarks[67]) + compute(landmarks[62],
                                                          landmarks[66]) + compute(landmarks[63], landmarks[65])

    down1 = compute(landmarks[60], landmarks[64])
    ratio1 = up1/(3.0*down1)

    yawning_threshold = 0.55

    if ratio1 >= yawning_threshold:
        return True
    else:
        return False


face_frame = np.zeros(1)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    face_frame = frame.copy()

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        yawnnn = yawning(landmarks)

        left_up = compute(landmarks[37], landmarks[41]) + \
            compute(landmarks[38], landmarks[40])
        left_down = compute(landmarks[36], landmarks[39])
        EAR_L = str(round(left_up/(2*left_down), 2))

        right_up = compute(landmarks[43], landmarks[47]) + \
            compute(landmarks[44], landmarks[46])
        right_down = compute(landmarks[45], landmarks[42])
        EAR_R = str(round(right_up/(2*right_down), 2))

        up_mouth = compute(landmarks[61], landmarks[67]) + compute(
            landmarks[62], landmarks[66]) + compute(landmarks[63], landmarks[65])
        down_mouth = compute(landmarks[60], landmarks[64])
        MAR_M = str(round(up_mouth/(3.0*down_mouth), 2))

        left_blink = blinked(landmarks[36], landmarks[37],
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        if yawnnn == True:
            yawn += 1
            if yawn > 6:
                status2 = "Yawning"
        else:
            status2 = ""

        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0

            if sleep > 6:
                status = "Sleeping"
                color = (255, 0, 0)
        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "About to"
                color = (0, 0, 255)
        else:

            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active"
                color = (0, 255, 0)

        cv2.putText(frame, status, (20, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 3)
        cv2.putText(frame, status2, (20, 130),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, color2, 3)
        if left_up > right_up:
            cv2.putText(frame, ear, (450, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, color2, 3)
            cv2.putText(frame, EAR_L, (550, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, color2, 3)
        else:
            cv2.putText(frame, ear, (450, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, color2, 3)
            cv2.putText(frame, EAR_R, (550, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, color2, 3)

        cv2.putText(frame, mar, (450, 140),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, color2, 2)

        cv2.putText(frame, MAR_M, (550, 140),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, color2, 2)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Frame", frame)
    cv2.imshow("Result of detector", face_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
