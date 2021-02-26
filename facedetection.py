import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

CameraCap = cv2.VideoCapture(0)

while True:
    _, frame = CameraCap.read()
    grayCap = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceCap = faceCascade.detectMultiScale(grayCap, scaleFactor=1.2, minNeighbors=4)
    print(np.mean(frame).round())


    for (x, y, w, h) in faceCap:
        print(x, y, w, h)
        # x --> right and left
        # y --> up and down
        # w --> forward and back
        # roiGray = grayCap[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "FACE", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("FRAME", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

CameraCap.release()
cv2.destroyAllWindows()
