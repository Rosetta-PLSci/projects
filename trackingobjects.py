import cv2
import numpy as np

VideoCap = cv2.VideoCapture(0)
tracker = cv2.TrackerMIL_create()
_, frame = VideoCap.read()
bbox = cv2.selectROI("TRACK", frame, False)
tracker.init(frame, bbox)


def drawbox(frame, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3, 1)


while True:
    _, frame = VideoCap.read()
    _, bbox = tracker.update(frame)

    if _:
        drawbox(frame, bbox)

    cv2.imshow("TRACK", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
