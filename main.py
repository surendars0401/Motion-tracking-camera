import cv2
import numpy as np


cap = cv2.VideoCapture(0)


ret, frame1 = cap.read()


prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)


mask = None

while True:

    ret, frame2 = cap.read()


    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)


    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)


    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])


    if mask is None:
        mask = np.zeros_like(frame2)


    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)


    cv2.imshow('Frame', frame2)
    cv2.imshow('Optical flow', rgb)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    prev_gray = gray


cap.release()
cv2.destroyAllWindows()
