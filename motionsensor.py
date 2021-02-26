import cv2
import numpy as np

mainVideo = cv2.VideoCapture("videoplayback.mp4")
# Video can be changeable
# Source can be a camera or a webcam

control, frame1 = mainVideo.read()
control, frame2 = mainVideo.read()

while mainVideo.isOpened():
    differenceVideo = cv2.absdiff(frame1, frame2)
    grayVideo = cv2.cvtColor(differenceVideo, cv2.COLOR_RGB2GRAY)
    blurVideo = cv2.GaussianBlur(grayVideo, (5, 5), 0)
    _, threshold = cv2.threshold(blurVideo, 20, 255, cv2.THRESH_BINARY)
    dilatedVideo = cv2.dilate(threshold, None, iterations=3)
    contours, _ = cv2.findContours(dilatedVideo, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, t, s) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 370:
            continue
        cv2.rectangle(frame1, (x, y), (x + t, y + s), (0, 255, 0), 2)
        cv2.putText(frame1, "STATUS: {}".format("MOVEMENT"), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    cv2.imshow("FEED", frame1)
    frame1 = frame2
    control, frame2 = mainVideo.read()

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
mainVideo.release()
