import cv2
import numpy as np

mainVideo = cv2.VideoCapture(0)

_, frame1 = mainVideo.read()
_, frame2 = mainVideo.read()


while mainVideo.isOpened():
    spec = frame1[120:148, 270:420]
    spec2 = frame2[120:148, 270:420]
    differenceVideo = cv2.absdiff(spec, spec2)
    grayVideo = cv2.cvtColor(differenceVideo, cv2.COLOR_RGB2GRAY)
    blurVideo = cv2.GaussianBlur(grayVideo, (5, 5), 0)
    _, threshold = cv2.threshold(blurVideo, 20, 255, cv2.THRESH_BINARY)
    dilatedVideo = cv2.dilate(threshold, None, iterations=3)
    contours, _ = cv2.findContours(dilatedVideo, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    for contour in contours:
        (x, y, t, s) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 600:
            continue
        cv2.rectangle(spec, (x, y), (x + t, y + s), (0, 255, 0), 2)
        #cv2.putText(spec, "STATUS: {}".format("MOVEMENT"), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print(1)

    cv2.imshow("FEED", frame1)
    cv2.imshow("EYE",spec)
    frame1 = frame2
    _, frame2 = mainVideo.read()

    if cv2.waitKey(35) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
mainVideo.release()
