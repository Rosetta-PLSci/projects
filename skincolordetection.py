import cv2
import numpy as np

VideoCap = cv2.VideoCapture("faces.mp4")
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

while True:
    _, frame = VideoCap.read()

    kernal = np.ones((7, 7), dtype="uint8")
    videoHSV = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    lowerforwhite = np.array([110, 110, 100])
    upperforwhite = np.array([113, 171, 185])
    lowerforblack = np.array([105, 116, 67])
    upperforblack = np.array([119, 250, 142])

    maskwhite = cv2.inRange(videoHSV, lowerforwhite, upperforwhite)
    maskblack = cv2.inRange(videoHSV, lowerforblack, upperforblack)
    whitedill = cv2.dilate(maskwhite, kernal)
    blackdill = cv2.dilate(maskblack, kernal)
    resultwhite = cv2.bitwise_and(frame, frame, mask=whitedill)
    resultblack = cv2.bitwise_and(frame, frame, mask=blackdill)
    contourswhite, _ = cv2.findContours(whitedill, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
    # THAT IS FOR WHITE SKIN
    #for contour in contourswhite:
        #area = cv2.contourArea(contour)

        #if area > 45000:
            #cv2.drawContours(frame, [contour], -1, (0, 0, 255), 1)
            #x, y, w, h = cv2.boundingRect(contour)
            #frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #cv2.putText(frame, "WHITE", (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 0, 0))

    contoursblack, _ = cv2.findContours(blackdill, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour2 in contoursblack:
        area = cv2.contourArea(contour2)

        if area > 15000:
            cv2.drawContours(frame, [contour2], -1, (0, 0, 255), 1)
            x, y, w, h = cv2.boundingRect(contour2)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "BLACK", (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 0, 0))

    cv2.imshow("LIVE", frame)

    if cv2.waitKey(140) & 0xFF == ord("q"):
        break

VideoCap.release()
cv2.destroyAllWindows()
