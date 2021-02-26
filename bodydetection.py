import cv2
import numpy as np
import matplotlib as pypl

CapVideo = cv2.VideoCapture(1)
bodyRec = cv2.CascadeClassifier("haarcascade_fullbody.xml")

while CapVideo.isOpened():
    _, frame = CapVideo.read()
    grayVideo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = bodyRec.detectMultiScale(grayVideo, 1.1, 3)

    for (x, y, t, s) in bodies:
        cv2.rectangle(frame, (x, y), (x + t, y + s), (0, 0, 255), 2)

    cv2.imshow("BODY", frame)
    if cv2.waitKey(55) & 0xFF == ord("q"):
        break

CapVideo.release()
cv2.destroyAllWindows()
