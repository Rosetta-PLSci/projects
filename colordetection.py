import cv2
import numpy as np

LiveCap = cv2.VideoCapture("crowded2.mp4")
# video is changeable
# source can be a camera or a webcam

_, frame = LiveCap.read()
_, frame2 = LiveCap.read()

while True:
    _, frame = LiveCap.read()
    kernal = np.ones((5, 5), dtype="uint8")
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    lower = np.array([1, 92, 47])
    upper = np.array([39, 255, 240])
    # for blue color
    # color is changeable

    mask = cv2.inRange(frameHSV, lower, upper)
    bluedill = cv2.dilate(mask, kernal)
    result = cv2.bitwise_and(frame, frame, mask=bluedill)
    contours, _ = cv2.findContours(bluedill, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10000:
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow("LIVE", frame)

    if cv2.waitKey(35) & 0xFF == ord("q"):
        break

LiveCap.release()
cv2.destroyAllWindows()
